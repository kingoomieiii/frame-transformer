sudo docker run \
    -it \
    --shm-size=10g \
    --gpus all \
    -v '/media/benjamin/Evo 870 SATA 1/cs2048_sr44100_hl1024_nf2048_of0_PRETRAINING':'/root/data/pretraining_p1' \
    -v '/media/benjamin/Evo 870 SATA 2/cs2048_sr44100_hl1024_nf2048_of0_PRETRAINING':'/root/data/pretraining_p2' \
    -v '/media/benjamin/External NVME 1/cs2048_sr44100_hl1024_nf2048_of0_PRETRAINING':'/root/data/pretraining_p3' \
    -v '/media/benjamin/External NVME 2/cs2048_sr44100_hl1024_nf2048_of0_PRETRAINING':'/root/data/pretraining_p4' \
    -v '/media/benjamin/Internal NVME 2TB/cs2048_sr44100_hl1024_nf2048_of0_PRETRAINING':'/root/data/pretraining_p5' \
    -v '/media/benjamin/Evo 870 SATA 2/cs2048_sr44100_hl1024_nf2048_of0':'/root/data/instruments_p1' \
    -v '/media/benjamin/External NVME 1/cs2048_sr44100_hl1024_nf2048_of0':'/root/data/instruments_p2' \
    -v '/media/benjamin/External NVME 2/cs2048_sr44100_hl1024_nf2048_of0':'/root/data/instruments_p3' \
    -v '/media/benjamin/Internal NVME 1/cs2048_sr44100_hl1024_nf2048_of0':'/root/data/instruments_p4' \
    -v '/media/benjamin/Internal NVME 2TB/cs2048_sr44100_hl1024_nf2048_of0':'/root/data/instruments_p5' \
    -v '/media/benjamin/Internal NVME 1/cs2048_sr44100_hl1024_nf2048_of0_VOCALS':'/root/data/vocals_p1' \
    -v '/media/benjamin/Internal NVME 2TB/cs2048_sr44100_hl1024_nf2048_of0_VOCALS':'/root/data/vocals_p2' \
    -v '/media/benjamin/Internal NVME 1/cs2048_sr44100_hl1024_nf2048_of0_VALIDATION':'/root/data/validation' \
    -v '/media/benjamin/External NVME 2/models':'/root/data/models' \
    --network=host \
    local_frame_transformer:latest