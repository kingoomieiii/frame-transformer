python3 \
    train-v1c.py \
        --gpu 0 \
        --batch_sizes 8 \
        --cropsizes 256 \
        --accumulation_steps 1 \
        --instrumental_lib '/root/data/instruments_p1|/root/data/instruments_p2|/root/data/instruments_p3|/root/data/instruments_p4|/root/data/instruments_p5' \
        --vocal_lib '/root/data/vocals_p1|/root/data/vocals_p2' \
        --validation_lib '/root/data/validation' \
        --model_dir '/root/data'