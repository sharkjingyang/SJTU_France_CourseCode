#!/bin/bash

echo Running on host: `hostname`
echo In directory: `pwd`
echo Starting on: `date`
echo SLURM_JOB_ID: $SLURM_JOB_ID
echo " "

###########################################################################################################

# DATASET="cifar10"
# CHANNEL_MULT="1 2 2"
# MODEL_CHANNELS="96"
# ATTN_RESOLUTIONS="16"
# LAYERS_PER_BLOCK="2"

# DATASET="mnist"
# CHANNEL_MULT="1 2 3 4"
# MODEL_CHANNELS="16"
# ATTN_RESOLUTIONS="0"
# LAYERS_PER_BLOCK="1"

DATASET="Oxford_flower"
CHANNEL_MULT="1 2 4 8"
MODEL_CHANNELS="64"
ATTN_RESOLUTIONS="16"
LAYERS_PER_BLOCK="2"

# DATASET="Scene"
# CHANNEL_MULT="1 2 4"
# MODEL_CHANNELS="64"
# ATTN_RESOLUTIONS="16"
# LAYERS_PER_BLOCK="2"


# training
python -u train_edm.py --dataset $DATASET \
        --channel_mult $CHANNEL_MULT --model_channels $MODEL_CHANNELS \
        --attn_resolutions $ATTN_RESOLUTIONS --layers_per_block $LAYERS_PER_BLOCK \
        --train_batch_size 64 --num_steps 100000 \
        --learning_rate 1e-4 --accumulation_steps 1 \
        --log_step 100 \
        --save_images_step 500 \
        --save_model_iters 1000 \


# ## sampling
# python -u sample_edm.py --dataset $DATASET \
#         --channel_mult $CHANNEL_MULT --model_channels $MODEL_CHANNELS \
#         --attn_resolutions $ATTN_RESOLUTIONS --layers_per_block $LAYERS_PER_BLOCK \
#         --sample_mode save \
#         --eval_batch_size 64 \
#         --model_paths \
#         --total_steps 40

# ## evaluate fid
# python -u sample_edm.py --dataset $DATASET \
#         --channel_mult $CHANNEL_MULT --model_channels $MODEL_CHANNELS \
#         --attn_resolutions $ATTN_RESOLUTIONS --layers_per_block $LAYERS_PER_BLOCK \
#         --sample_mode fid \
#         --num_fid_sample 5000 \
#         --fid_batch_size 512 \
#         --model_paths \
#         # --total_steps 40
       