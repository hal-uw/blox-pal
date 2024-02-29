export CUDA_VISIBLE_DEVICES=2,3
python -m torch.distributed.launch \
    --master_addr "127.0.0.1" \
    --master_port 6611 \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    main.py \
    --model-name 'resnet50' \
    --batch-size 128
