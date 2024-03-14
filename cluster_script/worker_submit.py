import os
import time
import subprocess
import pickle
import fcntl
import socket

import socket
import logging

# Define logging configurations
def setup_logging(rank):
    log_file = f'/scratch1/08503/rnjain/blox-pal/debug_worker_log_{rank}.txt'  # Adjust the path as needed
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get scheduler ip address
def get_ip_address():
    host_ip = socket.gethostbyname(socket.gethostname())
    logging.info("IP address of this node:", host_ip)
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
setup_logging(rank)

# Launch redis server on every node
logging.info("before start database")
subprocess.Popen('/scratch1/08503/rnjain/blox-pal/redis-7.2.4/src/redis-server &', shell=True)
logging.info("after start database")
time.sleep(3)
    
ip_addr = os.environ['SCHEDULER_NODE_IP']
logging.info(f"start node manager begin {ip_addr}") 
subprocess.Popen(f"nohup python -u /scratch1/08503/rnjain/worker_node/blox-pal/node_manager.py --ipaddr {ip_addr} --interface eno1 > /scratch1/08503/rnjain/worker_node/blox-pal/node_manager_stdout_{rank}.txt 2>&1 &", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
logging.info(f"start node manager end {ip_addr}")
os.environ['REGISTERED_WORKER_NODES'] = 1
time.sleep(30)

while True:
    pass
