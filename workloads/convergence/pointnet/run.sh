export CUDA_VISIBLE_DEVICES=3
for bs in 32 64 128
do
  python main.py \
        --batch-size $bs \
        --num-epochs 100
done