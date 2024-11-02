# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FreeSOLO/blob/main/LICENSE

export exp_name="d2fp_R101_bs64_72k_prior100_dec9"

python train_net.py \
	--dist-url tcp://127.0.0.1:$(( RANDOM % 1000 + 20100 )) \
	--num-gpus 2 \
	--config configs/lip/$exp_name.yaml \
	OUTPUT_DIR training_dir/lip/$exp_name \
