import sys
import time
import copy
import grpc
import json
import logging
import argparse
import pandas as pd
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
                        perfvar_arr = self._get_machine_slowdowns(gpu_uuid)
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


    def _get_machine_slowdowns(self, gpu_uuid):
        norm_perfvar = {}
        # Read profiles.json to get list of aggregated CSV names
        with open('profiles.json') as json_file:
            perfclasses = json.load(json_file)

        for key in perfclasses:
            df_new = pd.read_csv(perfclasses[key])

            # need to ensure that we sample at most 1 entry per unique GPU 
            df_sliced = df_new.loc[df_new.groupby('uuid')['perf'].idxmax()]

            median_value = df_sliced['perf'].median()

            df_sliced['perf'] = df_sliced['perf'].apply(lambda x: x / median_value)

            gpu_row = df_sliced[df_sliced['uuid'] == gpu_uuid]

            if not gpu_row.empty:
                norm_perfvar[key] = gpu_row['perf'].iloc[0]
            else:
                #randomly sample df_sliced['perf'] value from dataframe
                norm_perfvar[key] = random.choice(df_sliced['perf'])

        return norm_perfvar
    
    def get_gpus_by_job_id(self, job_id):
        # Filter the DataFrame based on JOB_ID and IN_USE
        filtered_df = self.gpu_df[(self.gpu_df['JOB_IDS'] == job_id) & (self.gpu_df['IN_USE'] == True)]
        gpu_ids = filtered_df['GPU_ID'].tolist()
        return gpu_ids