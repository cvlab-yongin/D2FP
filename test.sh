# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.

export exp_name="d2fp_R101_bs64_72k_dec9"
python train_net.py \
	--dist-url tcp://127.0.0.1:$(( RANDOM % 100 + RANDOM % 10 + 40000 )) \
	--num-gpus 2 \
	--config configs/lip/$exp_name.yaml \
	--eval-only \
	MODEL.WEIGHTS training_dir/lip/$exp_name/model_final.pth \

