set JOB_NAME=vocal_remover_job_01
set REGION=us-central1
set CONFIG=gcloud-p100x4.yaml

set NODES=4
set GPUS=1
set EPOCHS=1
set BATCH_SIZE=18
set NUM_ENCODERS=4
set NUM_DECODERS=4
set NUM_BANDS=8
set FEEDFORWARD_DIM=2048
set CHANNELS=8
set BIAS=True
set MIXED_PRECISION=True
set PROGRESS_BAR=True
set NUM_TRAINING_ITEMS=None
set NUM_WORKERS=12

set VANILLA=False
set LR=1e-3
set WARMUP_STEPS=4

gcloud ai-platform jobs submit training %JOB_NAME% --region %REGION% --config %CONFIG% -- --num_workers %NUM_WORKERS% --progress_bar %PROGRESS_BAR% --mixed_precision %MIXED_PRECISION% --nodes %NODES% --gpus %GPUS% --epochs %EPOCHS% --job_name %JOB_NAME% --batch_size %BATCH_SIZE% --num_encoders %NUM_ENCODERS% --num_decoders %NUM_DECODERS% --num_bands %NUM_BANDS% --feedforward_dim %FEEDFORWARD_DIM% --channels %CHANNELS% --bias %BIAS% --num_training_items %NUM_TRAINING_ITEMS%
gcloud ai-platform jobs describe %JOB_NAME%
gcloud ai-platform jobs stream-logs %JOB_NAME%