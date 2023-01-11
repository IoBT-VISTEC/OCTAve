import submitit


def job_distribute(jobfunc):
    """Job management decorator function. Expect jobfunc to utilize jdx, rdx via kwargs"""

    def wrapper(*args, **kwargs):
        # Get job with task rank.
        job_env = submitit.JobEnvironment()
        # Tell jobfunc about its job rank.
        jdx, rdx = job_env.global_rank, job_env.local_rank
        return jobfunc(*args, **kwargs, jdx=jdx, rdx=rdx)
    return wrapper