export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch \
    --master_addr "127.0.0.1" \
    --master_port 6666 \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    main.py \
    --model-name 'vgg19_bn' \
    --batch-size 32
