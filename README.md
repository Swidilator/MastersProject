# MastersProject
My 2019-2020 masters project

## Usage:
`./support_scripts/utils/ModelSettingsManager.py` includes the default values for these arguments.
```console
usage: Train.py [-h] [--model-conf-file MODEL_CONF_FILE] [--wandb] [--train TRAIN] [--starting-epoch STARTING_EPOCH]
                [--sample SAMPLE] [--sample-mode SAMPLE_MODE] [--sample-only]
                [--training-subset-size TRAINING_SUBSET_SIZE] [--base-model-save-dir BASE_MODEL_SAVE_DIR]
                [--model-save-prefix MODEL_SAVE_PREFIX] [--base-image-save-dir BASE_IMAGE_SAVE_DIR] [--cpu]
                [--gpu-no GPU_NO] [--save-every-num-epochs SAVE_EVERY_NUM_EPOCHS]
                [--load-saved-model LOAD_SAVED_MODEL] [--log-every-n-steps LOG_EVERY_N_STEPS] [--use-amp USE_AMP]
                [--num-data-workers NUM_DATA_WORKERS] [--flip-training-images] [--deterministic]
                [--max-run-hours MAX_RUN_HOURS] [--num-frames-per-training-video NUM_FRAMES_PER_TRAINING_VIDEO]
                [--num-frames-per-sampling-video NUM_FRAMES_PER_SAMPLING_VIDEO]
                [--prior-frame-seed-type PRIOR_FRAME_SEED_TYPE] [--video-frame-offset VIDEO_FRAME_OFFSET]
                [--use-mask-for-instances] [--use-saved-feature-encodings]
                [--sample-every-num-epochs SAMPLE_EVERY_NUM_EPOCHS] [--use-vid2vid-discriminators]
                model dataset_path input_image_height_width batch_size training_machine_name run_name

Masters model main file

positional arguments:
  model
  dataset_path
  input_image_height_width
  batch_size
  training_machine_name
  run_name

optional arguments:
  -h, --help            show this help message and exit
  --model-conf-file MODEL_CONF_FILE
  --wandb
  --train TRAIN
  --starting-epoch STARTING_EPOCH
  --sample SAMPLE
  --sample-mode SAMPLE_MODE
  --sample-only
  --training-subset-size TRAINING_SUBSET_SIZE
  --base-model-save-dir BASE_MODEL_SAVE_DIR
  --model-save-prefix MODEL_SAVE_PREFIX
  --base-image-save-dir BASE_IMAGE_SAVE_DIR
  --cpu
  --gpu-no GPU_NO
  --save-every-num-epochs SAVE_EVERY_NUM_EPOCHS
  --load-saved-model LOAD_SAVED_MODEL
  --log-every-n-steps LOG_EVERY_N_STEPS
  --use-amp USE_AMP
  --num-data-workers NUM_DATA_WORKERS
  --flip-training-images
  --deterministic
  --max-run-hours MAX_RUN_HOURS
  --num-frames-per-training-video NUM_FRAMES_PER_TRAINING_VIDEO
  --num-frames-per-sampling-video NUM_FRAMES_PER_SAMPLING_VIDEO
  --prior-frame-seed-type PRIOR_FRAME_SEED_TYPE
  --video-frame-offset VIDEO_FRAME_OFFSET
  --use-mask-for-instances
  --use-saved-feature-encodings
  --sample-every-num-epochs SAMPLE_EVERY_NUM_EPOCHS
  --use-vid2vid-discriminators
```

## Conda environment
The conda environment spec used to train these networks can be found in `./pytorch_environment.yml`.

## Dataset preparation
These were run inside docker to separate environments. A base pytorch image is used - https://gitlab.com/Swidilator/pytorchdockerfilebase.

DeeplabV3plus - https://gitlab.com/Swidilator/deeplabv3plus-pytorch

Detectron2 - https://gitlab.com/Swidilator/detectron2

## FVD
The modified FVD code and environment can be found at https://gitlab.com/Swidilator/frechet-video-distance-v2.