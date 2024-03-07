WEIGHT_PATH=weight/darknet53_448.weights
export CUDA_VISIBLE_DEVICES=2
python train.py --weight_path $WEIGHT_PATH
