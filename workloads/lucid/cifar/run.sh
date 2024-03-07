python -m torch.distributed.launch \
    --master_addr "127.0.0.1" \
    --master_port 6000 \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    profile_cifar_ddp.py
#    --master_addr "127.0.0.1" \
#    --master_port "7010" \
