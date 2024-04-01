import sys
import time
import copy
import grpc
import json
import logging
import argparse
import pandas as pd
import numpy as np
import time
import random
from concurrent import futures

from typing import Tuple, List

# import scheduler
# import placement

from blox_manager import BloxManager

# from profile_parsers import pparsers


class ClusterState(object):
    """
    Keep tracks of cluster state
    """

    def __init__(self, args: argparse.ArgumentParser) -> None:
        # self.blr = blox_resource_manager
        # keeps a map of nodes
        self.server_map = dict()
        # keeps the count of node
        self.node_counter = 0
        # number of GPUs
        self.gpu_number = 0
        # gpu dataframe for easy slicing
        self.gpu_df = pd.DataFrame(
            columns=[
                "GPU_ID",
                "GPU_UUID",
                "Local_GPU_ID",
                "Node_ID",
                "IP_addr",
                "IN_USE",
                "JOB_IDS",
                "PerfVar_classA",
                "PerfVar_classB",
                "PerfVar_classC",
                "PerfVar_classD",
            ]
        )
        self.time = 0
        self.cluster_stats = dict()
        random.seed(42)

    # def get_new_nodes(self):
    # """
    # Fetch any new nodes which have arrived at the scheduler
    # """
    # new_nodes = self.blr.rmserver.get_new_nodes()
    # return new_nodes

    def _add_new_machines(self, new_nodes: List[dict]) -> None:
        """
        Pops information of new machines and keep track of them

        new_node : list of new nodes registered with the resource manager
        """
        while True:
            try:
                node_info = new_nodes.pop(0)
                self.server_map[self.node_counter] = node_info
                numGPUs_on_node = node_info["numGPUs"]
                gpu_uuid_list = node_info["gpuUUIDs"].split("\n")
                assert (
                    len(gpu_uuid_list) == numGPUs_on_node
                ), f"GPU UUIDs {len(gpu_uuid_list)}  GPUs on node {numGPUs_on_node}"
                if numGPUs_on_node > 0:
                    gpuID_list = list()
                    for local_gpu_id, gpu_uuid in zip(
                        range(numGPUs_on_node), gpu_uuid_list
                    ):
                        # get per-class PerfVar value corresponding to gpu_uuid
                        # cluster
                        # perfvar_arr = self._get_machine_slowdowns(gpu_uuid)
                        # sim 
                        # perfvar_arr = self._get_sim_worst_slowdowns(self.gpu_number)
                        perfvar_arr = self._get_machine_slowdowns(self.gpu_number)
                        
                        gpuID_list.append(
                            {
                                "GPU_ID": self.gpu_number,
                                "Node_ID": self.node_counter,
                                "GPU_UUID": gpu_uuid,
                                "Local_GPU_ID": local_gpu_id,
                                "IP_addr": node_info["ipaddr"],
                                "IN_USE": False,
                                "JOB_IDS": None,
                                "PerfVar_classA": perfvar_arr["classA"],
                                "PerfVar_classB": perfvar_arr["classB"],
                                "PerfVar_classC": perfvar_arr["classC"],
                                "PerfVar_classD": perfvar_arr["classD"],
                            }
                        )
                        self.gpu_number += 1
                    self.gpu_df = self.gpu_df.append(gpuID_list)
                    self.node_counter += 1
            except IndexError:
                break

    def update(self, new_nodes):
        """
        Updates cluster state by fetching new nodes
        Args:
            None
        Returns:
            new_nodes : List of new nodes
        """
        # getting new updates
        # new_nodes = self.blr.rmserver.get_new_nodes()

        if len(new_nodes) > 0:
            self._add_new_machines(new_nodes)
        return new_nodes


    def _get_sim_slowdowns(self, gpu_uuid):
        norm_perfvar = {}
        num_replicas = self.gpu_number
        num_clusters = 8 #K value - tuning later

        # Read profiles.json to get list of aggregated CSV names
        with open('profiles.json') as json_file:
            perfclasses = json.load(json_file)

        df_cluster = pd.read_csv('aggregated_data/32gpu-variability.csv')    
        logging.debug("read in df_cluster {}".format(df_cluster.to_string()))
        for key in perfclasses:
            variability_val = df_cluster[df_cluster['GPU_ID'] == gpu_uuid][f"PerfVar_{key}"].values[0]
            norm_perfvar[key] =  variability_val

        return norm_perfvar
    

    def _get_sim_worst_slowdowns(self, gpu_id):
        norm_perfvar = {}

        # Read profiles.json to get list of aggregated CSV names
        with open('profiles.json') as json_file:
            perfclasses = json.load(json_file)

        # Read 32gpu-varaibility.csv to get mapping for gpu_id to cluster UUID 
        df = pd.read_csv("aggregated_data/32gpu-variability.csv")
        gpu_dict = df.set_index('GPU_ID')['GPU_UUID'].to_dict()
        logging.info(f"gpu id to uuid mapping dict {gpu_dict}")

        for key in perfclasses:
            df_sliced = pd.read_csv(perfclasses[key])

            median_value = df_sliced['perf'].median()
            df_sliced['perf'] = df_sliced['perf'].apply(lambda x: x / median_value)

            perf_lists = df_sliced.groupby('uuid')['perf'].apply(list) # same uuid can have multiple entries
            max_perfs = perf_lists.apply(lambda x: np.max(x))
            max_perfs_dict = max_perfs.to_dict()

            gpu_uuid = gpu_dict[gpu_id]
            if gpu_uuid in max_perfs_dict.keys():
                norm_perfvar[key] = max_perfs_dict[gpu_uuid]
            else:
                logging.info(f"Getting slowdown factors: randomly sampled for UUID {gpu_uuid}")
                #randomly sample df_sliced['perf'] value from dataframe
                norm_perfvar[key] = random.choice(df_sliced['perf'])

        return norm_perfvar        


    def _get_machine_slowdowns(self, gpu_uuid):
        norm_perfvar = {}
        # Read profiles.json to get list of aggregated CSV names
        with open('profiles.json') as json_file:
            perfclasses = json.load(json_file)

        for key in perfclasses:
            df_sliced = pd.read_csv(perfclasses[key])

            median_value = df_sliced['perf'].median()
            df_sliced['perf'] = df_sliced['perf'].apply(lambda x: x / median_value)

            perf_lists = df_sliced.groupby('uuid')['perf'].apply(list) # same uuid can have multiple entries
            median_perfs = perf_lists.apply(lambda x: np.median(x))
            median_perfs_dict = median_perfs.to_dict()
            
            if gpu_uuid in median_perfs_dict.keys():
                norm_perfvar[key] = median_perfs_dict[gpu_uuid]
            else:
                logging.info(f"Getting slowdown factors: randomly sampled for UUID {gpu_uuid}")
                #randomly sample df_sliced['perf'] value from dataframe
                norm_perfvar[key] = random.choice(df_sliced['perf'])

        return norm_perfvar
    
    def get_gpus_by_job_id(self, job_id):
        # Filter the DataFrame based on JOB_ID and IN_USE
        filtered_df = self.gpu_df[(self.gpu_df['JOB_IDS'] == job_id) & (self.gpu_df['IN_USE'] == True)]
        gpu_ids = filtered_df['GPU_ID'].tolist()
        return gpu_ids
