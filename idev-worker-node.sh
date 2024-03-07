#!/bin/bash

# Download and install redis
#wget https://github.com/redis/redis/archive/7.2.4.tar.gz
#tar -xvzf 7.2.4.tar.gz
#cd redis-7.2.4
#make

# Run the redis server
cd redis-7.2.4/src
./redis-server &

# Launch node manager with ip address of scheduler
interface_name=$(sed -n '1p' "$SCRATCH/scheduler-ip.txt")
ip_address=$(sed -n '2p' "$SCRATCH/scheduler-ip.txt" | awk '{print $1}')

echo "Interface name: $interface_name"
echo "IP Address (without subnet mask: $ip_address"

cd /scratch1/08503/rnjain/blox-pal
python node_manager.py --ipaddr "$ip_address" --interface "$interface_name"



