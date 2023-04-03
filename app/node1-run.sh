python -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr="10.0.0.34" --master_port=1234 \
    train-v1cgan-pre.py \
    --instrumental_lib '/root/data/instruments_p1|/root/data/instruments_p2|/root/data/instruments_p3|/root/data/instruments_p4|/root/data/instruments_p5' \
    --pretraining_lib '/root/data/pretraining_p1|/root/data/pretraining_p2|/root/data/pretraining_p3|/root/data/pretraining_p4|/root/data/pretraining_p5' \
    --vocal_lib '/root/data/vocals_p1|/root/data/vocals_p2' \
    --validation_lib '/root/data/validation' \
    --model_dir '/root/data/models'