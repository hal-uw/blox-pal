import pandas as pd
import copy
from typing import Tuple, List
import json
import random
import collections

class PackedNSPlacement(object):
    def __init__(self, args):
        pass

    @staticmethod
    def copy_arguments(function):
        def function_wrapper(
            self, job_state, cluster_state, new_job_schedule, **kwargs
        ):
            return function(
                self,
                copy.deepcopy(job_state.active_jobs),
                copy.deepcopy(new_job_schedule),
                copy.deepcopy(cluster_state.server_map),
                copy.deepcopy(cluster_state.gpu_df),
                **copy.deepcopy(kwargs),
            )

        return function_wrapper

    @copy_arguments.__func__
    def place(
        self,
        active_jobs: dict,
        new_job_schedule: dict,
        node_info: dict,
        gpu_df: pd.DataFrame,
        **kwargs,
    ) -> dict:
        """
        parses the sorted_jobs dictionary and calls relevant placement policy

        # CAUTION: This function makes in place changes to active jobs and
        # gpu_df

        """
        job_order = new_job_schedule["job_order"]
        scheduler = new_job_schedule.get("scheduler")
        jobs_to_terminate = list()
        job_to_launch = dict()
        launched_job_ids = list()
        # go over jobs in job order
        potential_preempt_dict = {}
        running_jobs = 0
        new_scheduled_jobs = 0
        jobs_to_schedule = 0

        # Non-sticky - clear all allocations for currently running jobs 
        # placement policy reapplied for each job in queue
        for item in job_order:
            jid, _ = item
            job = active_jobs[jid]

            if job["is_running"] == True:
                potential_preempt_dict[jid] = get_gpus_by_job_id(gpu_df, jid)
                delete_job_by_id(gpu_df, jid)
                jobs_to_terminate.append(jid)            

        for idx, job_id in enumerate(job_order):
            job_id, _ = job_id
            job = active_jobs[job_id]
            found = False

            place_consolidated = (
                job.get("placement_preference") == "consolidated"
            )

            # first checking if there are free GPUs
            free_gpus = find_free_GPUs(gpu_df)
            placement, found = self._consolidated_placement(job, free_gpus)

            if found:
                new_scheduled_jobs += 1
                job_to_launch[job_id] = placement
                mark_gpu_in_use(gpu_df, placement, job_id)
            else:
                print(f"New Jobs scheduled {new_scheduled_jobs}")
                print(f"Jobs previously running {running_jobs}")
                print(f"Jobs terminated {len(jobs_to_terminate)}")
                print(f"Jobs in queue {len(job_order)-idx}")
                break

        # if job id is a key in both potential_preempt_dict and job_to_launch
        # AND the value list is exactly the same in both dictionaries, 
        # remove jid from jobs_to_terminate
        for job_id in potential_preempt_dict.keys() & job_to_launch.keys():
            if sorted(potential_preempt_dict[job_id]) == sorted(job_to_launch[job_id]):
                jobs_to_terminate.remove(job_id)
        # with open("debug-default.log", "a") as file:
        #     file.write("==============================\n")
        #     for job_id, placement in job_to_launch.items():
        #         file.write(f"{job_id}, {active_jobs[job_id]['perfclass']}")
        #         json.dump(placement, file)
        #         file.write("\n")
        return (jobs_to_terminate, job_to_launch)

    def _consolidated_placement(
        self, job_param: dict, free_gpus: dict
    ) -> Tuple[list, bool]:
        """
        Find a consolidated placement
        Args:
        job_param: Job Param configuration
        free_gpus: Dict of free GPUs {node_id: [list of GPU IDs']}
        Returns:
        list of GPU IDs on which to place the job
        boolean indicating if we found placement
        """
        # with open("consolidated_placement.log", "a") as file:
        #     file.write(f"{job_param['job_id']},{job_param['num_GPUs']}, {job_param['wait_time']},{free_gpus}")
        #     file.write("\n")
        # if there is a machine with exact required GPUs
        numGPUs_needed = job_param["num_GPUs"]
        gpus_for_job = list()
        for node in free_gpus:
            if len(free_gpus[node]) == numGPUs_needed:
                # found a perfect match
                return (free_gpus[node], True)
        # if we don't find an exact match find a node more GPUs
        # find the mode with min more GPUs then needed
        min_more_GPUs = 256  # random large enough number
        node_with_min_more_GPUs = None
        for node in free_gpus:
            if len(free_gpus[node]) >= numGPUs_needed:
                # found a node with more GPUs then needed
                if min_more_GPUs > len(free_gpus[node]):
                    min_more_GPUs = len(free_gpus[node])
                    node_with_min_more_GPUs = node
        if node_with_min_more_GPUs is not None:
            # only extracting the GPUs we need
            return (free_gpus[node_with_min_more_GPUs][:numGPUs_needed], True)
        # else soft consolidated - pack on as few nodes as possible
        # first check if enough GPUs are available for this job
        all_free_gpus = sum(free_gpus.values(), [])

        if len(all_free_gpus) >= numGPUs_needed:
            node_gpu_counts = {k:len(list(v)) for k,v in free_gpus.items()}
            list_free_gpus = collections.Counter(node_gpu_counts)
            while len(gpus_for_job) < numGPUs_needed:
                node_idx, count = list_free_gpus.most_common(1)[0]
                num = min(count, numGPUs_needed-len(gpus_for_job))
                gpus_for_job.extend(free_gpus[node_idx][:num])
                list_free_gpus[node_idx] -= num
            return (gpus_for_job, True)

        # didn't find the requested number of GPUs
        return ([], False)

    def _scattered_placement(
        self, job_param: dict, free_gpus: dict
    ) -> Tuple[list, bool]:
        """
        Find placement without worrying about consolidation.
        Args:
        job_param: Job Param configuration
        free_gpus: Dict of free GPUs {node_id: [list of GPU IDs']}
        Returns:
        list of GPU IDs on which to place the job
        boolean indicating if we found placement
        """
        # with open("scattered-placement.log", "a") as file:
        #     file.write(f"{job_param['job_id']},{job_param['num_GPUs']}, {job_param['wait_time']},{free_gpus}")
        #     file.write("\n")
        numGPUs_needed = job_param["num_GPUs"]
        alloc_gpu = job_param["num_GPUs"]
        gpus_for_job = list()
        found = False
        for node in free_gpus:
            gpus_for_job.extend(free_gpus[node][:numGPUs_needed])
            numGPUs_needed = alloc_gpu - len(gpus_for_job)
            # if perfect match found
            if len(gpus_for_job) == alloc_gpu:
                found = True
                break
        if found:
            return (gpus_for_job, found)
        else:
            return ([], False)

    def _random_placement(
        self, job_param: dict, free_gpus: dict
    ) -> Tuple[list, bool]:
        """
        Find placement by randomly sampling as many free GPUs as the job requires
        Args:
        job_param: Job Param configuration
        free_gpus: Dict of free GPUs {node_id: [list of GPU IDs']}
        Returns:
        list of GPU IDs on which to place the job
        boolean indicating if we found placement
        """
        random_seed = 42 # any integer - fixed seed for run-to-run reproducibility
        # with open("scattered-placement.log", "a") as file:
        #     file.write(f"{job_param['job_id']},{job_param['num_GPUs']}, {job_param['wait_time']},{free_gpus}")
        #     file.write("\n")
        numGPUs_needed = job_param["num_GPUs"]
        gpus_for_job = list()
        found = False

        # Get list of free GPUs from free_gpus dictionary
        all_free_gpus = sum(free_gpus.values(), [])

        if len(all_free_gpus) > numGPUs_needed:
            # enough GPUs on cluster available to allocate to this job
            # randomly sample numGPUs_needed:
            gpus_for_job.extend(random.sample(all_free_gpus,numGPUs_needed))
            found = True

        if found:
            return (gpus_for_job, found)
        else:
            return ([], False)
# Gavel get job ids sorted by vals


def get_ids_sorted_by_priorities(priority_vals: dict) -> list:
    """
    Sorts the dict by value and return a sorted list in descending order of
    their priorities
    Args:
    priority_vals: key- job_id, vals- priority vals
    Returns:
    list of job ids sorted by their values
    """
    sorted_pairs = sorted(priority_vals.items(), key=lambda x: x[1], reverse=True)

    sorted_ids = [x for x, _ in sorted_pairs]
    return sorted_ids


# Pandas Utilities
def find_gpus_matching_JobID(job_id: int, gpu_df: pd.DataFrame) -> list:
    """
    Finds the GPU IDs which are running the given job id
    """
    return gpu_df.loc[gpu_df["JOB_IDS"] == job_id]["GPU_ID"].tolist()

def get_num_gpus_per_node(gpu_df: pd.DataFrame) -> int:
    """
    Returns number of GPUs on a node using gpu_df
    assumes all nodes have same NUM_GPUS_PER_NODE
    no error checking
    """
    return gpu_df["Node_ID"].value_counts().iloc[0]



# Find free GPUs


def find_free_GPUs(gpu_df: pd.DataFrame) -> dict:
    """
    Find the nodeID's which have free GPUs
    Args:
    gpu_df : DataFrame consisting of information about GPUs
    Returns:
    dict: {Node_ID: [list of free GPUs]}
    """
    return (
        gpu_df.loc[gpu_df["IN_USE"] == False]
        .groupby("Node_ID")["GPU_ID"]
        .apply(list)
        .to_dict()
    )


def find_free_GPUs_by_type(gpu_df: pd.DataFrame, gpu_type: str) -> dict:
    """
    Find free nodeID's which have free GPUs of specific type

    Args:
    gpu_df : DataFrame consiting the information about GPUs
    Returns:
    dict : {Node_ID : [list of free GPUs]}
    """
    return (
        gpu_df.loc[(gpu_df["IN_USE"] == False) & (gpu_df["GPU_type"] == gpu_type)]
        .groupby("Node_ID")["GPU_ID"]
        .apply(list)
        .to_dict()
    )


# Mark a GPU in use


def mark_gpu_in_use(gpu_df: pd.DataFrame, gpu_id: List[int], job_id: int) -> None:
    """
    Find the GPU ID and mark it in use. After deciding to schedule something on
    it.
    Args:
    gpu_df : DataFrame consisting of information about GPUs
    gpu_id : GPU to mark busy
    job_id: Job being scheduled on GPU with id=gpu_id

    Returns:
    None
    In place modifies the gpu_df
    """
    gpu_df.loc[gpu_df["GPU_ID"].isin(gpu_id), ["JOB_IDS", "IN_USE"]] = job_id, True
    return None


# Delete Job from data frame


def delete_job_by_id(gpu_df: pd.DataFrame, job_id: int) -> None:
    """
    Finds the job ID provided. Marks those jobs free and marks the GPU free to
    Args:
    gpu_df : DataFrame consisting of information about GPUs
    job_id : Job to delete

    Returns:
    None
    In place modifies the gpu_df
    """
    gpu_df.loc[gpu_df["JOB_IDS"] == job_id, ["JOB_IDS", "IN_USE"]] = None, False
    return None
