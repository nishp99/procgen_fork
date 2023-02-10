import submitit
import time

def add(a, b):
    print("Waiting 30 seconds...")
    time.sleep(2)
    print("Done.")
    return a + b

log_folder = "utils"
executor = submitit.AutoExecutor(folder=log_folder)
executor.update_parameters(timeout_min=4, slurm_partition="debug")

job = executor.submit(add, 5, 7)  # will compute add(5, 7)
print(job.job_id)  # ID of your job

output = job.result()  # waits for the submitted function to complete and returns its output
# if ever the job failed, job.result() will raise an error with the corresponding trace
assert output == 12  # 5 + 7 = 12...  your addition was computed in the cluster
print(f"Output was {output}.")

a = [1, 2, 3, 4]
b = [10, 20, 30, 40]
executor = submitit.AutoExecutor(folder=log_folder)

executor.update_parameters(slurm_array_parallelism=2, slurm_partition="debug")
jobs = executor.map_array(add, a, b)  # just a list of jobs

output = [job.result() for result in jobs]
print(f"Output was {output}.")