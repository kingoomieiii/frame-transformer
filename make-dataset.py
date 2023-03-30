from lib.dataset import make_validation_set, train_val_split, make_vocal_stems, make_dataset, make_mix_dataset

vocal_dataset = [
    # ("/media/ben/Evo 870 SATA 1/dataset/vocals_padding", "/media/ben/internal-nvme-b/")
]

validation = "/media/ben/Evo 870 SATA 1/dataset/validation"

cropsize = 256
hop_length = 1024
fft = 2048

dirs = [
    # ('/media/ben/Evo 870 SATA 1/dataset/instruments/training_p1', '/media/ben/internal-nvme-b/', True, False),
    # ('/media/ben/Evo 870 SATA 1/dataset/instruments/training_p10', '/home/ben/', True, False),
    # ('/media/ben/Evo 870 SATA 1/dataset/instruments/training_p2', '/media/ben/internal-nvme-b/', True, False),
    # ('/media/ben/Evo 870 SATA 1/dataset/instruments/training_p3', '/home/ben/', True, False),
    # ('/media/ben/Evo 870 SATA 1/dataset/instruments/training_p11', '/media/ben/internal-nvme-b/', True, False),
    # ('/media/ben/Evo 870 SATA 1/dataset/instruments/training_p4', '/home/ben/', True, False),
    # ('/media/ben/Evo 870 SATA 1/dataset/instruments/training_p15', '/media/ben/internal-nvme-b/', True, False),
    # ('/media/ben/Evo 870 SATA 1/dataset/instruments/training_p13', '/home/ben/', True, False),
    # ('/media/ben/Evo 870 SATA 1/dataset/instruments/training_p5', '/media/ben/internal-nvme-b/', True, False),
    # ('/media/ben/Evo 870 SATA 1/dataset/instruments/training_p7', '/home/ben/', True, False),
    # ('/media/ben/Evo 870 SATA 1/dataset/instruments/training_p12', '/media/ben/internal-nvme-b/', True, False),
    # ('/media/ben/Evo 870 SATA 1/dataset/instruments/training_p14', '/home/ben/', True, False),
    # ('/media/ben/Evo 870 SATA 1/dataset/instruments/training_p6', '/media/ben/internal-nvme-b/', True, False),
    # ('/media/ben/Evo 870 SATA 1/dataset/instruments/training_p9', '/home/ben/', True, False),
    # ('/media/ben/Evo 870 SATA 1/dataset/instruments/training_p16', '/media/ben/internal-nvme-b/', True, False),
]

for dir in dirs:
    train_filelist, _ = train_val_split(
        dataset_dir=dir[0],
        val_filelist=[],
        val_size=-1,
        train_size=-1,
        voxaug=dir[2])

    if not dir[2]:
        print(train_filelist)

    if not dir[3]:
        val_dataset = make_dataset(
            filelist=train_filelist,
            cropsize=cropsize,
            sr=44100,
            hop_length=hop_length,
            n_fft=fft,
            root=dir[1],
            is_validation=True,
            suffix='_PAIRS' if not dir[2] else '')
    else:
        val_dataset = make_mix_dataset(
            filelist=train_filelist,
            cropsize=cropsize,
            sr=44100,
            hop_length=hop_length,
            n_fft=fft,
            root=dir[1],
            is_validation=True)

val_filelist, _ = train_val_split(
    dataset_dir=validation,
    val_filelist=[],
    val_size=-1,
    train_size=-1,
    voxaug=False)

val_dataset = make_dataset(
    filelist=val_filelist,
    cropsize=cropsize,
    sr=44100,
    hop_length=hop_length,
    n_fft=fft,
    root="/media/ben/internal-nvme-b/",
    is_validation=True,
    suffix='_VALIDATION')

for dir in vocal_dataset:
    make_vocal_stems(dataset=dir[0], root=dir[1],
                     cropsize=cropsize, hop_length=hop_length, n_fft=fft)
