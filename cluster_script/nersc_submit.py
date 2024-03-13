import os
import time
import subprocess
import pickle
import fcntl
import socket

# we need to change following job id and job duration here.

def get_ip_address():
    # Fetch the hostname, which indirectly gives us the node name
    hostname = socket.gethostname()
    
    # Get the primary IP address associated with the hostname
    ip_address = socket.gethostbyname(hostname)
    
    return ip_address


def run_command(command):
    """Run a command and return the process if started successfully."""
    try:
        # Start the command as a subprocess
        process = subprocess.Popen(command, shell=True)
        # Optionally, wait for a moment to ensure the process starts
        time.sleep(1)
        print(f"Command '{command}' started successfully.")
        return process
    except Exception as e:
        print(f"Failed to start command '{command}': {e}")
        return None
    


rank = int(os.environ['SLURM_PROCID'])

if rank % 4 == 0:
    print("before start database")
    # with open(f'/pscratch/sd/s/songbian/redis_stdout_{rank}.txt', 'w') as out, open(f"/pscratch/sd/s/songbian/redis_stderr_{rank}.txt", "w") as err:
    subprocess.Popen('/redis-7.2.4/src/redis-server &', shell=True)
    print("after start database")
    time.sleep(3)
else:
    time.sleep(5)
    

if rank == 1:
    print("start scheduler begin")
    # with open(f'/pscratch/sd/s/songbian/scheduler_stdout_{rank}.txt', 'w') as out, open(f"/pscratch/sd/s/songbian/scheduler_stderr_{rank}.txt", "w") as err:
    subprocess.Popen('python /global/homes/s/songbian/blox-Megatron/megatron_scheduler_cluster.py --round-duration 60 --start-id-track 0 --stop-id-track 1 &', shell=True)
    print("start scheduler end")
    time.sleep(5)
else:
    time.sleep(10)


if rank % 4 == 2:
    ip_address = get_ip_address()
    print(f"start node manager begin {ip_address}") 
    master_addr = os.environ['MASTER_ADDR']
    print(f"master_addr: {master_addr}")
    # with open(f'/pscratch/sd/s/songbian/node_manager_stdout_{rank}.txt', 'w') as out, open(f"/pscratch/sd/s/songbian/node_manager_stderr_{rank}.txt", "w") as err:
    subprocess.Popen(f'nohup python -u /global/homes/s/songbian/blox-Megatron/node_manager.py --ipaddr {master_addr} --interface hsn1 > /pscratch/sd/s/songbian/node_manager_stdout_{rank}.txt 2>&1 &', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"start node manager end {ip_address}")
    time.sleep(30)
else:
    time.sleep(30)


if rank == 3:
    print("submit jobs begin") 
    # with open(f'/pscratch/sd/s/songbian/submit_stdout_{rank}.txt', 'w') as out, open(f"/pscratch/sd/s/songbian/submit_stderr_{rank}.txt", "w") as err:
    subprocess.Popen(f'nohup python -u /global/homes/s/songbian/Megatron-Resource/blox_exp/submit_modified.py > /pscratch/sd/s/songbian/submit_stdout_{rank}.txt 2>&1 &', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("submit jobs end") 
else:
    pass

while True:
    pass











