import os
import sys
import time
import json
import grpc
import argparse
import pandas as pd

from job import Job

sys.path.append(
    os.path.join(os.path.dirname(__file__), "../blox/deployment/grpc_stubs")
)
import rm_pb2 as rm_pb2
import rm_pb2_grpc as rm_pb2_grpc


def _get_perfclass(new_job):
    if "perfclass" in new_job and (new_job.get("perfclass")) is not None:
        return new_job["perfclass"]
    else:  
        model = new_job["job_model"]
        if model == "vgg19":
            return "classC"
        elif model == "resnet50": 
            return "classA"
        elif model == "DCGAN": 
            return "classB"
        elif model == "PointNet": 
            return "classD"
        elif model == "Bert-Large": 
            return "classB"
        elif model == "GPT2-Medium":
            return "classB"
        else:
            return "classA" #if unknown class, we will assume worst variabilty

def preprocess_job(new_job):
    new_job["exclusive"] = 1
    new_job["submit_time"] = new_job["job_arrival_time"]
    new_job["num_GPUs"] = new_job["job_gpu_demand"]
    new_job["perfclass"] = _get_perfclass(new_job)
    return new_job


def main(scheduler_ipaddr):
    """
    Parse the files from the given csv file and submit it to the central scheduler.
    Args:
        scheduler_ipaddr: IP-Address of the scheduler code.
    """
    df = pd.read_csv(
        filepath_or_buffer="/scratch1/08503/rnjain/blox-pal/blox_exp/workload/sia-160job-reduced.csv",
        dtype={
            "job_id": int,
            "num_gpus": int,
            "submit_time": int,
            "duration": int,
            "model": str,
            "batch_size": str,
        },
    )

    jobs = []
    for _, row in df.iterrows():
        job = Job(
            job_id=row.job_id,
            job_arrival_time=row.submit_time,
            job_gpu_demand=row.num_gpus,
            job_iteration_time=1,
            job_total_iteration=row.duration,
            job_model=row.model,
            batch_size=row.batch_size,
        )
        jobs.append(job)

    current_time = 0
    jcounter = 0
    current_job = None
    while True:
        if current_job is None:
            new_job = preprocess_job(jobs[jcounter].__dict__)
            current_job = new_job
            print(current_job)
        if current_job["submit_time"] <= current_time:
            # submit_job
            current_job["params_to_track"] = ["per_iter_time", "attained_service", "iter_num"]
            current_job["default_values"] = [0, 0, 0]
            current_job["simulation"] = False
            # NOTE: This seems to be taken care of
            # current_job["num_GPUs"] = current_job["num_gpus"]
            with grpc.insecure_channel(scheduler_ipaddr) as channel:
                stub = rm_pb2_grpc.RMServerStub(channel)
                response = stub.AcceptJob(
                    rm_pb2.JsonResponse(response=json.dumps(current_job))
                )
                print(f"Job Accepted: {response.value}")
            jcounter += 1
            if jcounter == len(jobs):
                break
            current_job = None
        else:
            time.sleep(1)
            current_time += 1


def parse_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser with arguments
    """
    parser.add_argument(
        "--scheduler-ipaddr",
        default="localhost:50051",
        type=str,
        help="Name of the scheduling strategy",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args(
        argparse.ArgumentParser(description="Arguments for Submitting jobs")
    )
    main(args.scheduler_ipaddr)
