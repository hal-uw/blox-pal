import os
import time
import subprocess


def run_command(command):
    """Run a command and return the process if started successfully."""
    try:
        # Start the command as a subprocess
        process = subprocess.Popen(command, shell=True)
        # Optionally, wait for a moment to ensure the process starts
        time.sleep(1)
        print(f"Command {command} started successfully.")
        return process
    except Exception as e:
        print(f"Failed to start command '{command}: {e}")
        return None


if int(os.environ['SLURM_PROCID']) == 0:
    print("before start database")
    process1 = run_command('/scratch1/08503/rnjain/blox-pal/redis-7.2.4/src/redis-server &')
    print("after start database")
    
    if process1 is not None:
        os.system('module load cuda')
        os.system('module load tacc-apptainer')
        os.system('module load gcc/11.2.0')

        print("start scheduler begin") 
        process2 = run_command('python /scratch1/08503/rnjain/blox-pal/las_scheduler_cluster.py --round-duration 30 --start-id-track 0 --stop-id-track 3 &')
        print("start scheduler end")
        if process2 is not None:
            print("start node manager begin") 
            master_addr = os.environ['MASTER_ADDR']
            print("master_addr: {master_addr}")
            process3 = run_command(f'python /scratch1/08503/rnjain/worker_node/blox-pal/node_manager.py --ipaddr {master_addr} --interface eno1 &')
            print("start node manager end") 

            print(f"submit jobs begin") 
            if process3 is not None:
                process4 = run_command('python /scratch1/08503/rnjain/worker_node/blox-pal/blox_exp/submit_modified.py &')

while True:
    pass



