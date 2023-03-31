from lib.dataset import make_validation_set, train_val_split, make_vocal_stems, make_dataset, make_mix_dataset

vocal_dataset = [
    # ("J://dataset/vocals", "C://"),
    # ("J://dataset/vocals2", "D://")
]

validation = [
    #("J://dataset/validation", "C://")
]

dirs = [
    # ('J:///dataset/instruments/training_p10', 'D://', True, False),
    # ('J:///dataset/instruments/training_p1', 'D://', True, False),
    # ('J:///dataset/instruments/training_p2', 'C://', True, False),
    # ('J:///dataset/instruments/training_p3', 'F://', True, False),
    # ('J:///dataset/instruments/training_p11', 'H://', True, False),
    # ('J:///dataset/instruments/training_p4', 'D://', True, False),
    # ('J:///dataset/instruments/training_p15', 'C://', True, False),
    # ('J:///dataset/instruments/training_p13', 'F://', True, False),
    # ('J:///dataset/instruments/training_p5', 'H://', True, False),
    # ('J:///dataset/instruments/training_p7', 'D://', True, False),
    # ('J:///dataset/instruments/training_p12', 'C://', True, False),
    # ('J:///dataset/instruments/training_p14', 'F://', True, False),
    # ('J:///dataset/instruments/training_p6', 'H://', True, False),
    # ('J:///dataset/instruments/training_p16', 'D://', True, False),
    # ('J:///dataset/instruments/training_p9', 'H://', True, False),
    # ('J:///dataset/instruments/training_p8', 'F://', True, False),
    # ('/media/ben/Evo 870 SATA 1/dataset/instruments/training_p1', '/media/ben/internal-nvme-b/', True, False),
    # ('/media/ben/Evo 870 SATA 1/dataset/instruments/training_p10', '/home/ben/', True, False),
    # ('/media/ben/Evo 870 SATA 1/dataset/instruments/training_p2', '/media/ben/internal-nvme-b/', True, False),
    # ('/media/ben/Evo 870 SATA 1/dataset/instruments/training_p3', '/home/ben/', True, False),
    # ('/media/ben/Evo 870 SATA 1/dataset/instruments/training_p11', '/media/ben/internal-nvme-b/', True, False),
    # ('/media/ben/Evo 870 SATA 1/dataset/instruments/training_p4', '/home/ben/', True, False),
]

pretraining_dirs = [
    ( "J://dataset/pretraining/pretraining_p1", "F://" ),
    ( "J://dataset/pairs/training_p5", "D://" ),
    ( "J://dataset/pairs/training_p6", "F://" ),
    ( "J://dataset/pairs/training_p7", "H://" ),
    ( "J://dataset/pairs/training_p12", "F://" ),
    ( "J://dataset/pairs/training_p13", "H://" ),
    # ( "J://dataset/pretraining/pretraining_p2", "F://" ),
    # ( "J://dataset/pretraining/pretraining_p3", "F://" ),
    # ( "J://dataset/pretraining/pretraining_p4", "F://" ),
    # ( "J://dataset/pretraining/pretraining_p5", "F://" ),
]

cropsize = 2048
hop_length = 1024
fft = 2048

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

for input_dir, output_dir in pretraining_dirs:
    train_filelist, _ = train_val_split(
        dataset_dir=input_dir,
        val_filelist=[],
        val_size=-1,
        train_size=-1,
        pretraining=True)
    
    val_dataset = make_dataset(
        filelist=train_filelist,
        cropsize=cropsize,
        sr=44100,
        hop_length=hop_length,
        n_fft=fft,
        root=output_dir,
        is_validation=True,
        suffix='_PRETRAINING')
        
for input_dir, output_dir in validation:
    val_filelist, _ = train_val_split(
        dataset_dir=input_dir,
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
        root=output_dir,
        is_validation=True,
        suffix='_VALIDATION')

for dir in vocal_dataset:
    make_vocal_stems(dataset=dir[0], root=dir[1],
                     cropsize=cropsize, hop_length=hop_length, n_fft=fft)
