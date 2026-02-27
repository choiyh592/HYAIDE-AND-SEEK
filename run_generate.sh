#!/bin/bash

# --- Configuration ---
TASK="generate" # choices: "generate", "interpret"
IMAGE_PATH="/home/yhchoi/Diffusion_Toy/corn_drawing.png" # path to your image
SAVE_PATH="/home/yhchoi/Diffusion_Toy/gen_exp" # path to save generated image
MODEL_ID="sana" # model name : case-insensitive but lowercase is recommended for experiment logging

# --- ODE & Latent Settings : Tune these ---
FW_METHOD="euler"
REV_METHOD="euler"
FW_STEPS=1
REV_STEPS=3
ALPHA=0.8 
INIT_TIME=0.0 # in range [0, 1)
IMAGE_INDEX=0
TARGET_SIZE=1024

# --- Execution ---
# Ensure the save directory exists
mkdir -p "$SAVE_PATH"

echo "Starting Hidden Pictures Pipeline: $TASK mode..."

python main.py \
    --task "$TASK" \
    --image_path "$IMAGE_PATH" \
    --save_path "$SAVE_PATH" \
    --model_id "$MODEL_ID" \
    --ode_method_fw "$FW_METHOD" \
    --ode_method_rev "$REV_METHOD" \
    --alpha "$ALPHA" \
    --num_steps_fw "$FW_STEPS" \
    --num_steps_rev "$REV_STEPS" \
    --init_time "$INIT_TIME" \
    --target_size "$TARGET_SIZE"

echo "Pipeline process complete." 