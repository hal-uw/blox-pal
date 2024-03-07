export CUDA_VISIBLE_DEVICES=2
for bs in 32 64
do
  python main.py \
        --batch-size $bs \
        --num-epochs 100
done