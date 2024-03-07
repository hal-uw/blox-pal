for i in 1
do
  export CUDA_VISIBLE_DEVICES=0
  python -m torch.distributed.launch \
      --master_addr "127.0.0.1" \
      --master_port 7322 \
      --nproc_per_node=1 \
      --nnodes=1 \
      --node_rank=0 \
      profile_pointnet_ddp.py \
      --batch-size 128
done
