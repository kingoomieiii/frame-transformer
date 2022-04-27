from lib.dataset import make_validation_set, train_val_split, make_vocal_stems, make_dataset

vocal_dataset = "E://Projects/vocal-remover-data/vocals"
dataset = "G://new-inst"
validation = "E://Projects/vocal-remover-data/validation"

cropsize = 512
hop_length = 1024
fft = 2048

dirs = [
    #("E://Projects/vocal-remover-data/training_p1", "C://Projects/vocal remover/vocal-remover-master/", True),
    #("E://Projects/vocal-remover-data/training_p2", "C://Projects/vocal remover/vocal-remover-master/", True),
    #("E://Projects/vocal-remover-data/training_p3", "", True),
    #("G://training_p1", 'C://', True),
    #("G://training_p3", 'C://', True),
    #("G://training_p4", 'G://', True),
    #("G://training_p5", 'G://', False),
    #("E://Projects/vocal-remover-data/training_p3", 'C://', True),
    #("E://Projects/vocal-remover-data/validation", 'C://', False),
]

for dir in dirs:
    train_filelist, _ = train_val_split(
        dataset_dir=dir[0],
        val_filelist=[],
        val_size=-1,
        train_size=-1,
        voxaug=False)

    print(train_filelist)

    val_dataset = make_dataset(
        filelist=train_filelist,
        cropsize=cropsize,
        sr=44100,
        hop_length=hop_length,
        n_fft=fft,
        root=dir[1],
        is_validation=True,
        suffix="PAIRS")

val_filelist, _ = train_val_split(
    dataset_dir=validation,
    val_filelist=[],
    val_size=-1,
    train_size=-1,
    voxaug=False)

val_dataset = make_validation_set(
    filelist=val_filelist,
    sr=44100,
    hop_length=hop_length,
    n_fft=fft)


#make_vocal_stems(dataset=vocal_dataset, root="G://", cropsize=cropsize, hop_length=hop_length, n_fft=fft)