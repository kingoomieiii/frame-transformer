export MASTER_ADDR="10.0.0.34"
export MASTER_PORT=1234

torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
            train-v1cgan-pre.py \
                --gpu 0 \
                --batch_sizes 4 \
                --cropsizes 256 \
                --accumulation_steps 1 \
                --instrumental_lib '/root/data/instruments_p1|/root/data/instruments_p2' \
                --pretraining_lib '/root/data/pretraining_p1' \
                --vocal_lib '/root/data/vocals_p1|/root/data/vocals_p2' \
                --validation_lib '/root/data/validation' \
                --model_dir '/root/data/models' \
                --world_rank 0 \
                --distributed true