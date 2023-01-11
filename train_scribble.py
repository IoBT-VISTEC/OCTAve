import os
import sys
import signal
import time

from loguru import logger
logger.debug("Logger initialized")

import click
import submitit
from tqdm import tqdm

from utils.cluster.slurm import OverrideSlurmExecutor as SlurmExecutor


@click.group()
@click.option('--wandb-online/--wandb-offline', default=False)
def cli(wandb_online: bool):
    if not wandb_online:
        logger.info(f'WANDB_MODE: {"online" if wandb_online else "offline"}')
        os.environ['WANDB_MODE'] = 'offline'

def interrupt_handler(signal, frame):
    if input("Abort? (All running job will not be stop!) [yY/nN]:") in ('y', 'Y'):
        sys.exit(1)
    else:
        print("Resume...")

# Register handler
signal.signal(signal.SIGINT, interrupt_handler)


@cli.command()
@click.option('--train', is_flag=True, help="Run training")
@click.option('--prepare-data', is_flag=True, help="Prepare data")
@click.option('--seed', type=int, default=50, help='Set global seed')
@click.option('--folds', type=int, default=5)
@click.option('--scribble-presence-ratio', type=float, default=1.0)
@click.option('--num-epochs', type=int, default=300)
@click.option('--experiment-start-index', type=int, default=0)
@click.option('--experiment-end-index', type=int, default=None)
@click.option('--mem-gb', type=int, default=12)
@click.option('--timeout-min', default=240)
@click.option('--force-slurm', is_flag=True, help='Forced slurm job executor.')
@click.option('--slurm-partition', type=str, default='gpu-cluster')
@click.option('--exclude-node', type=str, default='')
@click.option('--concurrency', type=int, default=4)
@click.option('--nodes', '-N', type=int, default=1, help='Number of nodes per job')
@click.option('--num-gpus-per-node', '-NG', type=int, default=1, help='Number of gpus per node')
@click.option('--log-dir', default='./slurm_explogs/')
@click.option('--base_logging_frequency', type=int, default=10)
@click.option('--dryrun', is_flag=True, help='Dry running')
def run_octa500(
    train, prepare_data, seed, folds, scribble_presence_ratio, num_epochs, experiment_start_index, experiment_end_index,
    mem_gb, timeout_min, force_slurm, slurm_partition, exclude_node, concurrency, nodes, num_gpus_per_node, log_dir, base_logging_frequency, dryrun):

    logger.info(f"LOGDIR: {log_dir}")

    if not (train or prepare_data):
        click.echo("`train` or `prepare-data` must be specified.")
        sys.exit(0)

    from pytorch_lightning import seed_everything
    seed_everything(seed)
    from runsets.octa500.experiments.experiments import ScribbleNetOCTA500ExperimentConfig, submit_scribblenet_experiment

    # Summon Execution Engine
    logger.info("Initialized the Executor")
    if force_slurm:
        logger.info("Forced Slurm Executor")
        executor = SlurmExecutor(log_dir)
        added_params = {}
        if num_gpus_per_node > 1:
            added_params['ntasks_per_node'] = num_gpus_per_node
            logger.info(f'Adding `ntasks_per_node`: {num_gpus_per_node}')
        if exclude_node != '':
            added_params['exclude'] = exclude_node
            logger.info(f'Excluding: {exclude_node}')
        executor.update_parameters(
            mem=f'{mem_gb}GB',
            partition=slurm_partition,
            gpus_per_node=num_gpus_per_node,
            nodes=nodes,
            time=timeout_min,
            **added_params
        )
    else:
        executor = submitit.AutoExecutor(log_dir)
        executor.update_parameters(
            mem_gb=mem_gb,
            slurm_partition=slurm_partition,
            gpus_per_node=num_gpus_per_node,
            tasks_per_node=num_gpus_per_node,
            nodes=nodes,
            timeout_min=timeout_min,
        )
    num_jobs_per_submit = nodes * num_gpus_per_node
    experimentations = ScribbleNetOCTA500ExperimentConfig(
        n_folds=folds, num_epochs=num_epochs,
        start_index=experiment_start_index, end_index=experiment_end_index,
        get_parallel_job=num_jobs_per_submit, base_logging_frequency=base_logging_frequency, scribble_presence_ratio=scribble_presence_ratio)
    logger.info("Constructed experimentation configuration")
    

    if prepare_data:
        logger.info('Constructing dataset...')
        prepare_job = executor.submit(experimentations.prepare_data, experimentations.dataset)
        while not prepare_job.done():
            logger.info(f"STDOUT: {prepare_job.stdout()}")
            logger.info(f"STDERR: {prepare_job.stderr()}")
            time.sleep(1)
        logger.info('Construction completed.')

    if train:
        logger.info(f"Starting experiment from index {experiment_start_index}")
        logger.info(f"Experiment(s): {len(experimentations)}")

        job_queue = []
        _stop_flag = False
        iterate_experiment = iter(experimentations)
        logger.info(f'Experiment List:')
        logger.info(f'{list(map(lambda exp: exp.experiment_name, experimentations.experiments))}')
        with tqdm(total=len(experimentations), desc='Experimentation Progress', unit='experiment') as prog_bar:
            queue_bar = tqdm(desc='Job Queue', total=concurrency, unit='job')
            while not _stop_flag:
                # Submitting job untill queue full or no job left.
                if not _stop_flag and len(job_queue) < concurrency:
                    retry = 5
                    success = False
                    try:
                        experiment = next(iterate_experiment)
                        while (not success) and retry > 0:
                            try:
                                job = executor.submit(submit_scribblenet_experiment, experiment, dry_run=dryrun)
                                success = True
                            except Exception as e:
                                logger.error(f'Failed to submit job due to following exception {e}')
                                logger.error(f'retry left: {retry}')
                                retry -= 1
                                time.sleep(60)
                        job_queue.append(job)
                        queue_bar.update(1)
                        time.sleep(1)
                    except StopIteration:
                        _stop_flag = True
                # Iterating object for deletion.
                if len(job_queue) > 0:
                    freezed_len = len(job_queue)
                    for _ in range(freezed_len):
                        job = job_queue.pop()
                        if not dryrun:
                            if job.done():
                                del job
                                prog_bar.update(num_jobs_per_submit)
                                queue_bar.update(-1)
                            else:
                                job_queue.append(job)
                        else:
                                prog_bar.update(num_jobs_per_submit)
                                queue_bar.update(-1)
                time.sleep(10)
            # Residual
            while len(job_queue) > 0:
                freezed_len = len(job_queue)
                for _ in range(freezed_len):
                    job = job_queue.pop()
                    if job.done():
                        del job
                        prog_bar.update(1)
                        queue_bar.update(-1)
                    else:
                        job_queue.append(job)
                if not dryrun:
                    time.sleep(10)
            queue_bar.close()
            logger.info("All experiments has concluded...")


if __name__ == '__main__':
    cli()
