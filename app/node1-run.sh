export MASTER_ADDR="10.0.0.34"
export MASTER_PORT=1234

python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
            train-v1cgan-pre.py \
                --gpu 0 \
                --instrumental_lib '/root/data/instruments_p1|/root/data/instruments_p2|/root/data/instruments_p3|/root/data/instruments_p4|/root/data/instruments_p5' \
                --pretraining_lib '/root/data/pretraining_p1|/root/data/pretraining_p2|/root/data/pretraining_p3|/root/data/pretraining_p4|/root/data/pretraining_p5' \
                --vocal_lib '/root/data/vocals_p1|/root/data/vocals_p2' \
                --validation_lib '/root/data/validation' \
                --model_dir '/root/data/models' \
                --world_rank 1 \
                --distributed true