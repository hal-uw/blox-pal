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

# Get scheduler ip address
def get_ip_address():
    host_ip = socket.gethostbyname(socket.gethostname())
    logging.info(f"IP address of this node: {host_ip}")
    return host_ip 

def run_command(command):
    """Run a command and return the process if started successfully."""
    try:
        # Start the command as a subprocess
        process = subprocess.Popen(command, shell=True)
        # Optionally, wait for a moment to ensure the process starts
        time.sleep(1)
        logging.info(f"Command '{command}' started successfully.")
        return process
    except Exception as e:
        logging.info(f"Failed to start command '{command}': {e}")
        return None
    
rank = int(os.environ['SLURM_PROCID'])
node_id = os.environ['SLURM_NODEID']
setup_logging(node_id, rank)
ip_addr = os.environ['SCHEDULER_NODE_IP_ADDR']

# Launch redis server on every node
if rank % 4 == 0: 
    logging.info("before start database")
    subprocess.Popen('/scratch1/08503/rnjain/blox-pal/redis-7.2.4/src/redis-server &', shell=True)
    logging.info("after start database")
    time.sleep(3)
else:
    time.sleep(5)
    

if rank == 1:
    node_id = os.environ['SLURM_NODEID']
    rank = int(os.environ['SLURM_PROCID'])
    setup_logging(node_id, rank)
    logging.info("start scheduler begin")
    subprocess.Popen('nohup python -u /scratch1/08503/rnjain/blox-pal/fifo_scheduler_cluster.py --round-duration 30 --start-id-track 0 --stop-id-track 2 &', shell=True)
    logging.info("start scheduler end")
    time.sleep(5)
else:
    time.sleep(10)


if rank % 4 == 2:
    node_id = os.environ['SLURM_NODEID']
    rank = int(os.environ['SLURM_PROCID'])
    setup_logging(node_id, rank)
    logging.info(f"Running node manager on node ID {node_id} and rank {rank} with IP address of scheduler {ip_addr}")
    subprocess.Popen(f"nohup python -u /scratch1/08503/rnjain/worker_node/blox-pal/node_manager.py --ipaddr {ip_addr} --interface eno1 > /scratch1/08503/rnjain/worker_node/blox-pal/node_manager_stdout_{rank}.txt 2>&1 &", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    logging.info("Launched node_manager")
    time.sleep(30)
else:
    time.sleep(30)


if rank == 3:
    node_id = os.environ['SLURM_NODEID']
    rank = int(os.environ['SLURM_PROCID'])
    setup_logging(node_id, rank)
    logging.info("sleep to give time for nodes to register with scheduler")
    time.sleep(30)
    logging.info("submit jobs begin") 
    subprocess.Popen(f'nohup python -u /scratch1/08503/rnjain/worker_node/blox-pal/blox_exp/submit_modified.py > /scratch1/08503/rnjain/worker_node/blox-pal/submit_stdout_{rank}.txt 2>&1 &', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    logging.info("submit jobs end") 
else:
    pass

while True:
    pass
