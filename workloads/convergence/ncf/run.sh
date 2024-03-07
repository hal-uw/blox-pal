export CUDA_VISIBLE_DEVICES=2
for bs in 256 512 1024 2048 4096
do
  python main.py \
        --batch-size $bs \
        --num-epochs 10
done