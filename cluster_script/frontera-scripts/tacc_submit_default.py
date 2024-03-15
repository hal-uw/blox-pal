import os
import time
import subprocess
import pickle
import fcntl
import socket
import logging

# Define logging configurations
def setup_logging(node_id, rank):
    log_file = f'/scratch1/08503/rnjain/blox-pal/debug_tacc_submit_log_{node_id}_{rank}.txt'
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

rank = int(os.environ['SLURM_PROCID'])
node_id = os.environ['SLURM_NODEID']
setup_logging(node_id, rank)
ip_addr = os.environ['SCHEDULER_NODE_IP_ADDR']


# Launch redis server on every node
logging.info("before start database")
subprocess.Popen('/scratch1/08503/rnjain/blox-pal/redis-7.2.4/src/redis-server &', shell=True)
logging.info("after start database")

if rank == 0:
    logging.info("Launching scheduler: check /scratch1/08503/rnjain/blox-pal/scheduler_stdout.txt")
    subprocess.Popen('nohup python -u /scratch1/08503/rnjain/blox-pal/fifo_scheduler_cluster.py --round-duration 30 --start-id-track 0 --stop-id-track 2 > /scratch1/08503/rnjain/blox-pal/scheduler_stdout.txt 2>&1 &', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) 
    time.sleep(3)
else:
    time.sleep(5)
    
logging.info(f"Running node manager on node ID {node_id} and rank {rank} with {ip_addr} IP address of scheduler")
subprocess.Popen(f"nohup python -u /scratch1/08503/rnjain/worker_node/blox-pal/node_manager.py --ipaddr {ip_addr} --interface eno1 > /scratch1/08503/rnjain/worker_node/blox-pal/node_manager_stdout_{rank}.txt 2>&1 &", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
logging.info("Launched node_manager")
logging.info("sleep to give time for nodes to register with scheduler")
time.sleep(30)

if rank == 0:
    logging.info("submit jobs begin") 
    subprocess.Popen(f'nohup python -u /scratch1/08503/rnjain/worker_node/blox-pal/blox_exp/submit_modified.py > /scratch1/08503/rnjain/worker_node/blox-pal/submit_stdout_{rank}.txt 2>&1 &', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    logging.info("submit jobs end") 

while True:
    pass
