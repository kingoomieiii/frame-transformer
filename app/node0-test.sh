python3 \
    train-v1cgan-pre.py \
        --gpu 0 \
        --instrumental_lib '/root/data/instruments_p1|/root/data/instruments_p2' \
        --pretraining_lib '/root/data/pretraining_p1' \
        --vocal_lib '/root/data/vocals_p1|/root/data/vocals_p2' \
        --validation_lib '/root/data/validation' \
        --model_dir '/root/data/models'