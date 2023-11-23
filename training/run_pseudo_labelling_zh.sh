#!/usr/bin/env bash

# TODO: add in zh-CN as well as zh-TW (targeting mandarin so exclude zh-HK)
accelerate launch run_pseudo_labelling.py \
  --model_name_or_path "openai/whisper-large-v3" \
  --dataset_name "mozilla-foundation/common_voice_13_0" \
  --dataset_config_name "zh-TW" \
  --dataset_split_name "train+validation+test" \
  --text_column_name "sentence" \
  --id_column_name "path" \
  --output_dir "./common_voice_13_0_hi_pseudo_labelled" \
  --wandb_project "distil-whisper-labelling" \
  --per_device_eval_batch_size 16 \
  --dtype "bfloat16" \
  --dataloader_num_workers 4 \
  --preprocessing_num_workers 4 \
  --logging_steps 500 \
  --max_label_length 128 \
  --report_to "wandb" \
  --language "zh" \
  --task "transcribe" \
  --return_timestamps \
  --attn_type "flash_attn" \
  --streaming False \
  --generation_num_beams 1 \
  --decode_token_ids False \
  --push_to_hub
