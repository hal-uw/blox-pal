import os
import sys
import time
import copy
import grpc
import json
import logging
import argparse
import pandas as pd
import time
from concurrent import futures

from typing import Tuple, List

# import scheduler
# import placement

# from profile_parsers import pparsers
sys.path.append(os.path.dirname(__file__))
import deployment.grpc_server_rm as rm_serve
import deployment.grpc_client_rm as rm_client

# from cluster_state import ClusterState
# from job_state import JobState

class BloxManager(object):
    """
    Implements the Blox bookkeeping interface, including accepting, scheduling and running jobs
    """

    def __init__(self, args: argparse.ArgumentParser) -> None:
        self.args = args
        self.scheduler_name = args.scheduler_name
        self.placement_name = args.placement_name
        self.acceptance_policy = args.acceptance_policy
        self.exp_prefix = args.exp_prefix
        self.load = args.load
        self.round_duration = args.round_duration
        self.comm_node_manager = rm_client.ResourceManagerComm(
            node_manager_port=args.node_manager_port
        )
        self.priority_thresh = 3600 * 1000  # above this we will have priority thresh
        self.server, self.rmserver = launch_server(
            rm_server_rpc_port=args.central_scheduler_port,
            simulator_rpc_port=args.simulator_rpc_port,
        )
        self.time = 0
        self.first_submit_time = None
        self.last_round_time = 0
        self.terminate = False
        return None

    def reset(self, args: argparse.ArgumentParser) -> None:
        """
        Change some runtime parameters post launch
        """
        self.args = args
        self.scheduler_name = args.scheduler_name
        self.placement_name = args.placement_name
        self.acceptance_policy = args.acceptance_policy
        self.exp_prefix = args.exp_prefix
        self.load = args.load
        self.round_duration = args.round_duration
        # self.comm_node_manager = rm_client.ResourceManagerComm()
        # self.priority_thresh = 3600 * 1000  # above this we will have priority thresh
        # self.server, self.rmserver = launch_server()
        self.time = 0
        self.terminate = False
        return None

    def terminate_server(self):
        """
        Shut down grpc server
        """
        # print("In terminate")
        self.server.stop(0)

    def update_cluster(self, cluster_state):
        """
        Update cluster state

        Args:
            cluster_state - Cluster State Object
        Returns:
            new_nodes - New nodes added in this cluster
        """
        # new_nodes = cluster_state.update()
        new_nodes = self.rmserver.get_new_nodes()
        cluster_state.update(new_nodes)
        return new_nodes

    def _get_avg_responsiveness(self, time_dict):
        values = list(time_dict.values())
        if len(values) > 0:
            return sum(values) * 1.0 / len(values)
        else:
            return 0

    def _get_avg_jct(self, time_dict):
        """
        Fetch the avg jct from the dict
        """
        values = list(time_dict.values())
        count = 0
        jct_time = 0
        for v in values:
            jct_time += v[1] - v[0]
            count += 1

        return jct_time / count

    def _get_99th_jct(self, time_dict):
        values = list(time_dict.values())
        jct_list = []
        for v in values:
            jct_list.append(v[1] - v[0])
        sorted_jct_list = sorted(jct_list)
        return sorted_jct_list[int(len(sorted_jct_list) * 0.99)]

    def update_metrics(self, cluster_state, job_state):
        """
        Perform metric collection also prunes the jobs.
        Args:
            cluster_state: Cluster State object
            job_state: Job State object

        Return:
            None
        """
        job_id_to_fetch = list()
        ipaddress_to_fetch_from = list()
        if_simulation = list()

        for jid in job_state.active_jobs:
            if job_state.active_jobs[jid]["is_running"] == True:
                job_id_to_fetch.append(jid)
                if_simulation.append(job_state.active_jobs[jid]["simulation"])
                ipaddress_to_fetch_from.append(job_state.active_jobs[jid]["rank_0_ip"])
        metric_data = self.comm_node_manager.get_metrics(
            job_id_to_fetch,
            ipaddress_to_fetch_from,
            if_simulation,
            self.round_duration,
            job_state.active_jobs,
        )
        job_state.update_metrics(metric_data, self.round_duration)
        # prune jobs which have been completed
        print("Metric Data {}".format(metric_data))
        logging.info(f"Metric Data {metric_data}")

        jid_to_terminate = list()
        for jid in job_state.active_jobs:
            if job_state.active_jobs[jid]["is_running"] == True:
                # Add job_exit for jobs whose num_iter updated above exceeds trace value
                if "iter_num" in metric_data[jid]:
                    if (
                        metric_data[jid]["iter_num"]
                        >= job_state.active_jobs[jid]["job_total_iteration"]
                    ):
                        logging.info(f"Marking {jid} for job_exit - Completed {metric_data[jid]['iter_num']} iterations")
                        job_state.active_jobs[jid]["tracked_metrics"]["job_exit"] = True 
                        
                # Get completion stats - separate check because its possible job_exit is true withut iter_num condition
                if jid in job_state.active_jobs:
                    if "tracked_metrics" in job_state.active_jobs[jid]:
                        if "job_exit" in job_state.active_jobs[jid]["tracked_metrics"]:
                            if (
                                job_state.active_jobs.get(jid)
                                .get("tracked_metrics")
                                .get("job_exit")
                                == True
                            ):
                                logging.info(f"Gathering completion stats {jid} = {time.time()} - {job_state.active_jobs[jid]['submit_time']}")
                                # TODO: put a condition to check if need
                                # plotting
                                if (
                                    jid >= job_state.job_ids_to_track[0]
                                    and jid <= job_state.job_ids_to_track[-1]
                                ):
                                    job_state.job_completion_stats[jid] = [
                                        job_state.active_jobs[jid]["submit_time"],
                                        time.time(),
                                    ]

                                    job_state.job_runtime_stats[jid] = copy.deepcopy(
                                        job_state.active_jobs[jid]
                                    )

                                jid_to_terminate.append(jid)
                                # delete GPU utilization
                                _free_gpu_by_jobid(jid, cluster_state.gpu_df)
                                # log the finished jobs
                                job_state.finished_job[jid] = 1

        # additional information for logging responsiveness
        for jid in job_state.active_jobs:
            if job_state.active_jobs[jid]["is_running"] == True:
                if jid in job_state.active_jobs:
                    if "job_launched_first_time" in job_state.active_jobs[jid]:
                        if (
                            job_state.active_jobs.get(jid).get(
                                "job_launched_first_time"
                            )
                            == True
                        ):
                            # TODO: put a condition to check if need
                            # plotting
                            if (
                                jid >= job_state.job_ids_to_track[0]
                                and jid <= job_state.job_ids_to_track[-1]
                            ):
                                # log the exit
                                job_state.job_responsiveness_stats[jid] = (
                                    self.last_round_time
                                    - job_state.active_jobs[jid]["submit_time"]
                                )
            # elif job_state.active_jobs[jid]["is_running"] == False:
            #     # The following information is used for allox+
            #     if job_state.active_jobs[jid]["simulation"]:
            #         job_state.active_jobs[jid]["wait_time"] += self.round_duration
            #     else:
            #         job_state.active_jobs[jid]["wait_time"] += time.time() - self.last_round_time
            # if the job is preempted
            if job_state.active_jobs[jid]["is_running"] == False:
                if "previously_launched" in job_state.active_jobs[jid]:
                    if (
                            job_state.active_jobs.get(jid).get(
                                "previously_launched"
                            )
                            == True
                    ):
                        if (
                                jid >= job_state.job_ids_to_track[0]
                                and jid <= job_state.job_ids_to_track[-1]
                        ):
                            # log the exit
                            if job_state.active_jobs[jid]["simulation"]:
                                job_state.job_responsiveness_stats[jid] += self.round_duration
                            else:
                                job_state.job_responsiveness_stats[jid] += time.time() - self.last_round_time

        terminate_list_id = list()
        terminate_ipaddr = list()
        terminate_simulation = list()
        print(f"jid_to_terminate: {jid_to_terminate}")
        filtered_gpudf = cluster_state.gpu_df[['Node_ID','GPU_ID','Local_GPU_ID','IN_USE','JOB_IDS']]
        print("State of cluster in update_metrics {}".format(filtered_gpudf))

        logging.info(f"In Blox Manager: update_metrics : {jid_to_terminate} \n")
        print("Job IDs to terminate {}".format(jobs_to_terminate))
        for jid in jid_to_terminate:
            # find ipaddresses for corresponding jobs to terminate
            running_ipddr = list(
                set(_find_ipaddr_by_job_ids(jid, cluster_state.gpu_df))
            )
            print(f"running_ipddr: {running_ipddr}")
            terminate_list_id.extend([jid] * len(running_ipddr))
            terminate_ipaddr.extend(running_ipddr)
            terminate_simulation.extend(
                [job_state.active_jobs[jid]["simulation"]] * len(running_ipddr)
            )
            # mark the job that is running is false
            job_state.active_jobs[jid]["is_running"] = False
            job_state.active_jobs[jid]["rank_0_ip"] = None
            # the job was suspended
            job_state.active_jobs[jid]["suspended"] = 1
            job_state.active_jobs.pop(jid)
            # delete GPU utilization
            _free_gpu_by_jobid(jid, cluster_state.gpu_df)
            # log the finished jobs
            job_state.finished_job[jid] = 1

        print(f"Calling blox_manager terminate_jobs: {terminate_list_id}, terminate ipaddr {terminate_ipaddr}, terminate_simulation {terminate_simulation}\n")
        logging.info(f"Calling blox_manager terminate_jobs: {terminate_list_id}, terminate ipaddr {terminate_ipaddr}, terminate_simulation {terminate_simulation}\n")
        self.comm_node_manager.terminate_jobs(
            terminate_list_id, terminate_ipaddr, terminate_simulation
        )
        filtered_columns = cluster_state.gpu_df[['GPU_ID', 'Local_GPU_ID', 'IN_USE', 'JOB_IDS']]
        print("Cluster State after pruning jobs based on runtime {}".format(filtered_columns))

        # update cluster use
        total_jobs, jobs_in_queue, jobs_running = _get_jobs_status(job_state)

        # gpu utilization

        free_gpus = len(
            cluster_state.gpu_df[cluster_state.gpu_df["IN_USE"] == False][
                "GPU_ID"
            ].tolist()
        )

        gpu_demand = 0
        running_jobs_dict = {}
        for jid in job_state.active_jobs:
            gpu_demand += job_state.active_jobs[jid]["num_GPUs"]
            if job_state.active_jobs[jid]["is_running"] == True:
                running_jobs_dict[jid] = cluster_state.get_gpus_by_job_id(jid)

        cluster_state.cluster_stats[self.time] = {
            "total_jobs": total_jobs,
            "jobs_in_queue": jobs_in_queue,
            "jobs_running": jobs_running,
            "free_gpus": free_gpus,
            "gpu_demand": gpu_demand,
            "allocation_dict": running_jobs_dict,
        }

        if any(jid in job_state.finished_job for jid in job_state.job_ids_to_track):
            avg_jct = self._get_avg_jct(job_state.job_completion_stats)
            jct_99th = self._get_99th_jct(job_state.job_completion_stats)
            avg_responsiveness = self._get_avg_responsiveness(job_state.job_responsiveness_stats)
            if self.first_submit_time is not None:
                makespan = time.time() - self.first_submit_time
            else:
                makespan = 0
            if not os.path.exists("results"):
                os.makedirs("results")
            with open("results/terminate_stat.txt", 'w') as f:
                f.write(
                    f"Scheduler: {self.scheduler_name}\n"
                    f"Placement: {self.placement_name}\n"
                    f"Acceptance Policy: {self.acceptance_policy}\n"
                    f"Avg JCT: {round(avg_jct, 2)}\n"
                    f"99th JCT: {round(jct_99th, 2)}\n"
                    f"Avg responsiveness: {round(avg_responsiveness, 2)}\n"
                    f"Makespan: {makespan}"
                )
            print(
                f"Scheduler: {self.scheduler_name}, "
                f"Placement: {self.placement_name}, "
                f"Acceptance Policy: {self.acceptance_policy}, "
                f"Avg JCT: {round(avg_jct, 2)}, "
                f"99th JCT: {round(jct_99th, 2)}, "
                f"Avg responsiveness: {round(avg_responsiveness, 2)}, "
                f"Makespan: {makespan}"
            )
            print(f"simulator_time: {self.time}")

            results_path = "/scratch1/08503/rnjain/blox-pal/results"
            with open(
                f"{results_path}/{self.exp_prefix}_{job_state.job_ids_to_track[0]}_{job_state.job_ids_to_track[-1]}_{self.scheduler_name}_{self.acceptance_policy}_{self.placement_name}_load_{self.load}_job_stats.json",
                "w",
            ) as fopen:
                # fopen.write(json.dumps(self.job_completion_stats))
                avg_jct = self._get_avg_jct(job_state.job_completion_stats)
                print(
                    f"Scheduler: {self.scheduler_name}, Acceptance Policy: {self.acceptance_policy}, Load: {self.load}, Avg JCT {avg_jct}"
                )
                json.dump(job_state.job_completion_stats, fopen)
            with open(
                f"{results_path}/{self.exp_prefix}_{job_state.job_ids_to_track[0]}_{job_state.job_ids_to_track[-1]}_{self.scheduler_name}_{self.acceptance_policy}_{self.placement_name}_load_{self.load}_cluster_stats.json",
                "w",
            ) as fopen:
                # fopen.write(json.dumps(self.cluster_stats))
                json.dump(cluster_state.cluster_stats, fopen)
            # sys.exit(0)
            with open(
                f"{results_path}/{self.exp_prefix}_{job_state.job_ids_to_track[0]}_{job_state.job_ids_to_track[-1]}_{self.scheduler_name}_{self.acceptance_policy}_{self.placement_name}_load_{self.load}_run_time_stats.json",
                "w",
            ) as fopen:
                # fopen.write(json.dumps(self.cluster_stats))
                json.dump(job_state.job_runtime_stats, fopen)

            # with open(
            #     f"{results_path}/{self.exp_prefix}_{job_state.job_ids_to_track[0]}_{job_state.job_ids_to_track[-1]}_{self.scheduler_name}_{self.acceptance_policy}_{self.placement_name}_load_{self.load}_responsivness.json",
            #     "w",
            # ) as fopen:
            #     # fopen.write(json.dumps(self.cluster_stats))
            #     avg_responsiveness = self._get_avg_jct(
            #         job_state.job_responsiveness_stats
            #     )
            #     print(
            #         f"Scheduler: {self.scheduler_name}, Acceptance Policy: {self.acceptance_policy}, Load: {self.load}, Avg responsiveness {avg_responsiveness}"
            #     )
            #     json.dump(job_state.job_responsiveness_stats, fopen)
            with open(
                f"{results_path}/{self.exp_prefix}_{job_state.job_ids_to_track[0]}_{job_state.job_ids_to_track[-1]}_{self.scheduler_name}_{self.acceptance_policy}_{self.placement_name}_load_{self.load}_custom_metrics.json",
                "w",
            ) as fopen:
                # fopen.write(json.dumps(self.cluster_stats))
                json.dump(job_state.custom_metrics, fopen)

        if all(jid in job_state.finished_job for jid in job_state.job_ids_to_track):
            self.terminate = True

        return None

    def pop_wait_queue(self, is_simulation: bool):
        """
        Get jobs which have arrived during the previous scheduling round
        """

        if is_simulation:
            # get jobs for simulation
            new_jobs = self.rmserver.get_jobs_sim(self.time)

        else:
            # get jobs for real cluster
            new_jobs = self.rmserver.get_new_jobs()
        return new_jobs

    def exec_jobs(
        self,
        jobs_to_launch: dict,
        jobs_to_terminate: list,
        cluster_state,
        active_jobs,
    ) -> None:
        """
        First terminates the jobs. Then marks the jobs to launch.
        Args:
            jobs_to_launch: {Job_ID: [GPUs to launch]}
            jobs_to_terminate : List of Job IDs to Terminate
            cluster_state : ClusterState class
            active_jobs: JobState
        Return:
          None
        """
        setup_logging("exec_jobs")
        terminate_list_id = list()
        terminate_ipaddr = list()
        terminate_simulation = list()
        logging.info(f"In blox_manager.exec_jobs - jobs_to_terminate = {jobs_to_terminate}")
        print("Job IDs to terminate {}".format(jobs_to_terminate))
        for jid in jobs_to_terminate:
            # find ipaddresses for corresponding jobs to terminate
            running_ipddr = list(
                set(_find_ipaddr_by_job_ids(jid, cluster_state.gpu_df))
            )
            terminate_list_id.extend([jid] * len(running_ipddr))
            terminate_ipaddr.extend(running_ipddr)
            terminate_simulation.extend(
                [active_jobs.active_jobs[jid]["simulation"]] * len(running_ipddr)
            )
            # mark the job that is running is false
            active_jobs.active_jobs[jid]["is_running"] = False
            active_jobs.active_jobs[jid]["rank_0_ip"] = None
            # the job was suspended
            active_jobs.active_jobs[jid]["suspended"] = 1
            # mark corresponding GPUs on which the jobs are running as
            # available
            _free_gpu_by_jobid(jid, cluster_state.gpu_df)
        filtered_columns = cluster_state.gpu_df[['GPU_ID', 'Local_GPU_ID', 'IN_USE', 'JOB_IDS']]
        print("Cluster state in exec_jobs before jobs_to_launch {}".format(filtered_columns))
        logging.info(f"List of jobs to terminate: {terminate_list_id}")
        self.comm_node_manager.terminate_jobs(
            terminate_list_id, terminate_ipaddr, terminate_simulation
        )

        # jobs terminated
        print(f"jobs_to_launch: {jobs_to_launch}")
        for jid in jobs_to_launch:
            gpus_to_launch = jobs_to_launch[jid]
            ipaddress_to_launch = _find_ipaddr_by_gpu_ids(
                gpus_to_launch, cluster_state.gpu_df
            )
            print(f"gpus_to_launch: {gpus_to_launch}")
            print(f"ipaddress_to_launch: {ipaddress_to_launch}")
            local_gpu_ids = _find_local_gpu_id(gpus_to_launch, cluster_state.gpu_df)
            print(f"local_gpu_ids: {local_gpu_ids}")
            self.comm_node_manager.launch_job(
                jid, active_jobs.active_jobs[jid], local_gpu_ids, ipaddress_to_launch
            )
            active_jobs.active_jobs[jid]["is_running"] = True
            active_jobs.active_jobs[jid]["rank_0_ip"] = list(set(ipaddress_to_launch))

            if "suspended" in active_jobs.active_jobs[jid]:
                active_jobs.active_jobs[jid]["suspended"] = 0
            _mark_gpu_in_use_by_gpu_id(gpus_to_launch, jid, cluster_state.gpu_df)

        # update the time for training

        for jid in active_jobs.active_jobs:
            if jid in jobs_to_terminate:
                active_jobs.active_jobs[jid]["time_since_scheduled"] = 0
            elif jid in jobs_to_launch:
                active_jobs.active_jobs[jid]["time_since_scheduled"] = 0
            elif active_jobs.active_jobs[jid]["is_running"]:
                active_jobs.active_jobs[jid]["time_since_scheduled"] = 0
            else:
                active_jobs.active_jobs[jid][
                    "time_since_scheduled"
                ] += self.round_duration
                if (
                    active_jobs.active_jobs[jid]["time_since_scheduled"]
                    >= self.priority_thresh
                ):
                    active_jobs.active_jobs[jid]["job_priority"] = 1


def _get_jobs_status(job_state) -> Tuple[int]:
    """
    Get number of jobs running, jobs in queue and total jobs
    """
    total_jobs = len(job_state.active_jobs.keys())
    jobs_in_queue = 0
    jobs_running = 0
    for jid in job_state.active_jobs:
        if job_state.active_jobs[jid]["is_running"]:
            jobs_running += 1
        if not job_state.active_jobs[jid]["is_running"]:
            jobs_in_queue += 1
    return (total_jobs, jobs_in_queue, jobs_running)


# NOTE: Utilities for querying the GPU DF
def _find_ipaddr_by_job_ids(job_id: str, gpu_df: pd.DataFrame) -> List[str]:
    """
    Given a jobID finds the ip-addresses on which the job runs.
    Args:
        job_id: ID of the job to find corresponding ipaddress
    Returns:
        List of IP addresses on which the job is running
    """
    return gpu_df[gpu_df["JOB_IDS"] == job_id]["IP_addr"].tolist()


def _find_ipaddr_by_gpu_ids(gpu_ids: List[int], gpu_df: pd.DataFrame) -> List[str]:
    """
    Return the IP address for given GPU IDs

    Args:
        gpu_ids: GPU ids to search
    Returns:
        List of IP addresses for corresponding gpu_ids
    """

    ipaddress_to_launch = list()
    for gid in gpu_ids:
        gid_ipaddr = gpu_df[gpu_df["GPU_ID"] == gid]["IP_addr"].tolist()
        assert len(gid_ipaddr) == 1, "Multiple IP addr for same GPU, something wrong"

        ipaddress_to_launch.extend(gid_ipaddr)
    return ipaddress_to_launch


def _free_gpu_by_jobid(job_id: int, gpu_df: pd.DataFrame) -> None:
    """
    Marks the corresponding GPU free for a given job ID
    Args:
        job_id: ID of the job to terminate
    """
    gpu_df.loc[gpu_df["JOB_IDS"] == job_id, ["JOB_IDS", "IN_USE"]] = (
        None,
        False,
    )
    return None


def _mark_gpu_in_use_by_gpu_id(
    gpu_id_list: List[int], job_id: int, gpu_df: pd.DataFrame
) -> None:
    """
    Marks the corresponding GPU in use for a given job ID
    Args:
        gpu_id_list : List of GPU ID's to terminate
        job_id: ID of the job to terminate
    """
    gpu_df.loc[gpu_df["GPU_ID"].isin(gpu_id_list), ["JOB_IDS", "IN_USE"]] = (
        job_id,
        True,
    )
    return None


def _find_local_gpu_id(global_gpu_ids: List[int], gpu_df: pd.DataFrame) -> List[int]:
    """
    Given a list of Global GPU ID's find the corresponding local GPU id's

    Args:
        global_gpu_ids: Global GPU ID's
    Returns:
        local_gpu_ids: Local GPU id's corresponding to that value
    """
    local_gpu_id = list()
    # TODO: Get rid of this for loop using .isin
    for gid in global_gpu_ids:
        lgid = gpu_df[gpu_df["GPU_ID"] == gid]["Local_GPU_ID"].tolist()

        assert (
            len(lgid) == 1
        ), "Multiple Local GPUs for same global GPU ID, something wrong"

        local_gpu_id.extend(lgid)

    return local_gpu_id


# print(metric_data)
# utility functions
def launch_server(
    rm_server_rpc_port: int, simulator_rpc_port: int
) -> Tuple[grpc.Server, rm_serve.RMServer]:
    """
    Launches GRPC server and returns the server object
    Args:
        None
    Returns:
        server : GRPC server object
        rmserver : The class object to work with rmserver
    """
    rmserver = rm_serve.RMServer(simulator_rpc_port=simulator_rpc_port)
    server = rm_serve.start_server(rmserver, rm_server_rpc_port=rm_server_rpc_port)
    print("Server started")
    return server, rmserver
