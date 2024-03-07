for i in 1
do
  export CUDA_VISIBLE_DEVICES=0
  python profile_rl_walker.py --batch-size 128
done
