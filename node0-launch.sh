sudo docker run \
    -it \
    --shm-size=10g \
    --gpus all \
    -v '/media/ben/external-nvme-1/cs2048_sr44100_hl1024_nf2048_of0_PRETRAINING':'/root/data/pretraining_p1' \
    -v '/media/ben/internal-nvme-b/cs2048_sr44100_hl1024_nf2048_of0':'/root/data/instruments_p1' \
    -v '/home/ben/cs2048_sr44100_hl1024_nf2048_of0':'/root/data/instruments_p2' \
    -v '/media/ben/internal-nvme-b/cs2048_sr44100_hl1024_nf2048_of0_VOCALS':'/root/data/vocals_p1' \
    -v '/home/ben/cs2048_sr44100_hl1024_nf2048_of0_VOCALS':'/root/data/vocals_p2' \
    -v '/media/ben/internal-nvme-b/cs2048_sr44100_hl1024_nf2048_of0_VALIDATION':'/root/data/validation' \
    -v '/media/ben/internal-nvme-b/models':'/root/data/models' \
    --network=host \
    local_frame_transformer:latest