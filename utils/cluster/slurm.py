from pathlib import Path
import shlex
import typing as tp
import warnings

from submitit import SlurmExecutor
from submitit.core import utils


class OverrideSlurmExecutor(SlurmExecutor):

    def _make_submission_file_text(self, command: str, uid: str) -> str:
        return _make_sbatch_string(command=command, folder=self.folder, **self.parameters)


# Override function due to the prohibition on `ntasks_per_node` configuration.
def _make_sbatch_string(
    command: str,
    folder: tp.Union[str, Path],
    job_name: str = "submitit",
    partition: str = None,
    time: int = 5,
    nodes: int = 1,
    ntasks_per_node: int = 1,
    cpus_per_task: tp.Optional[int] = None,
    cpus_per_gpu: tp.Optional[int] = None,
    num_gpus: tp.Optional[int] = None,  # legacy
    gpus_per_node: tp.Optional[int] = None,
    gpus_per_task: tp.Optional[int] = None,
    qos: tp.Optional[str] = None,  # quality of service
    setup: tp.Optional[tp.List[str]] = None,
    mem: tp.Optional[str] = None,
    mem_per_gpu: tp.Optional[str] = None,
    mem_per_cpu: tp.Optional[str] = None,
    signal_delay_s: int = 90,
    comment: str = "",
    constraint: str = "",
    exclude: str = "",
    gres: str = "",
    exclusive: tp.Union[bool, str] = False,
    array_parallelism: int = 256,
    wckey: str = "submitit",
    stderr_to_stdout: bool = False,
    map_count: tp.Optional[int] = None,  # used internally
    additional_parameters: tp.Optional[tp.Dict[str, tp.Any]] = None,
) -> str:
    """Creates the content of an sbatch file with provided parameters

    Parameters
    ----------
    See slurm sbatch documentation for most parameters:
    https://slurm.schedmd.com/sbatch.html

    Below are the parameters that differ from slurm documentation:

    folder: str/Path
        folder where print logs and error logs will be written
    signal_delay_s: int
        delay between the kill signal and the actual kill of the slurm job.
    setup: list
        a list of command to run in sbatch befure running srun
    map_size: int
        number of simultaneous map/array jobs allowed
    additional_parameters: dict
        Forces any parameter to a given value in sbatch. This can be useful
        to add parameters which are not currently available in submitit.
        Eg: {"mail-user": "blublu@fb.com", "mail-type": "BEGIN"}

    Raises
    ------
    ValueError
        In case an erroneous keyword argument is added, a list of all eligible parameters
        is printed, with their default values
    """
    nonslurm = [
        "nonslurm",
        "folder",
        "command",
        "map_count",
        "array_parallelism",
        "additional_parameters",
        "setup",
        "signal_delay_s",
        "stderr_to_stdout",
    ]
    parameters = {k: v for k, v in locals().items() if v and v is not None and k not in nonslurm}
    # rename and reformat parameters
    parameters["signal"] = f"USR1@{signal_delay_s}"
    if job_name:
        parameters["job_name"] = utils.sanitize(job_name)
    if comment:
        parameters["comment"] = utils.sanitize(comment, only_alphanum=False)
    if num_gpus is not None:
        warnings.warn(
            '"num_gpus" is deprecated, please use "gpus_per_node" instead (overwritting with num_gpus)'
        )
        parameters["gpus_per_node"] = parameters.pop("num_gpus", 0)
    if "cpus_per_gpu" in parameters and "gpus_per_task" not in parameters:
        warnings.warn('"cpus_per_gpu" requires to set "gpus_per_task" to work (and not "gpus_per_node")')
    if ntasks_per_node != 1:
        warnings.warn("`ntasks_per_node` is not set to 1 and will be forcefully set to 1 due to constraint applied.")
    del parameters['ntasks_per_node']
    # add necessary parameters
    paths = utils.JobPaths(folder=folder)
    stdout = shlex.quote(str(paths.stdout))
    stderr = shlex.quote(str(paths.stderr))
    # Job arrays will write files in the form  <ARRAY_ID>_<ARRAY_TASK_ID>_<TASK_ID>
    if map_count is not None:
        assert isinstance(map_count, int) and map_count
        parameters["array"] = f"0-{map_count - 1}%{min(map_count, array_parallelism)}"
        stdout = stdout.replace("%j", "%A_%a")
        stderr = stderr.replace("%j", "%A_%a")
    parameters["output"] = stdout.replace("%t", "0")
    if not stderr_to_stdout:
        parameters["error"] = stderr.replace("%t", "0")
    parameters["open-mode"] = "append"
    if additional_parameters is not None:
        parameters.update(additional_parameters)
    # now create
    lines = ["#!/bin/bash", "", "# Parameters"]
    lines += [
        "#SBATCH --{}{}".format(k.replace("_", "-"), "" if parameters[k] is True else f"={parameters[k]}")
        for k in sorted(parameters)
    ]
    # environment setup:
    if setup is not None:
        lines += ["", "# setup"] + setup
    # commandline (this will run the function and args specified in the file provided as argument)
    # We pass --output and --error here, because the SBATCH command doesn't work as expected with a filename pattern
    stderr_flag = "" if stderr_to_stdout else f"--error {stderr}"
    lines += [
        "",
        "# command",
        "export SUBMITIT_EXECUTOR=slurm",
        f"srun --output {stdout} {stderr_flag} --unbuffered {command}\n",
    ]
    return "\n".join(lines)
