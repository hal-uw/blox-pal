#!/bin/bash

# Download and install redis
wget https://github.com/redis/redis/archive/7.2.4.tar.gz
tar -xvzf 7.2.4.tar.gz
pushd redis-7.2.4
make

# Run the redis server
cd src
./redis-server &

# Clean up
rm 7.2.4.tar.gz

echo "Running ip a and filtering output for interface eno1"
ip_output=$(ip a | grep "eno1")

# Extract IP address and interface name using awk
ip_address=$(echo "$ip_output" | awk '/inet / {print $2}')
interface_name=$(echo "$ip_output" | awk '{print $2}')

# Print the results
echo "IP address for interface $interface_name: $ip_address"

# Save the information to a file
echo "$interface_name" > "$SCRATCH/scheduler-ip.txt"
echo "$ip_address" >> "$SCRATCH/scheduler-ip.txt"

echo "Information saved to $SCRATCH/scheduler-ip.txt"

cd $SCRATCH

echo "Laucnhing Scheduler"
# Launch scheduler
# python las_scheduler_cluster.py --round-duration 30 --start-id-track 0 --stop-id-track 10 &







