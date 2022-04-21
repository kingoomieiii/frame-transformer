from lib.dataset import train_val_split, make_vocal_stems, make_dataset

selected_validation = [ 
    "Thousands Carybdea",
    "Black Heart",
    "Miocene - Pliocene",
    "165 mixture",
    "567 MIX"
]

vocal_dataset = "G://new-vocals"
dataset = "G://new-inst"

cropsize = 256
hop_length = 1024
fft = 2048

dirs = [
    #("E://Projects/vocal-remover-data/training_p1", "C://Projects/vocal remover/vocal-remover-master/", True),
    #("E://Projects/vocal-remover-data/training_p2", "C://Projects/vocal remover/vocal-remover-master/", True),
    #("E://Projects/vocal-remover-data/training_p3", "", True),
    ("G://training_p3", 'C://', True),
    #("G://training_p5", 'G://', False),
]

for dir in dirs:
    train_filelist, _ = train_val_split(
        dataset_dir=dir[0],
        val_filelist=[],
        val_size=-1,
        train_size=-1,
        voxaug=dir[2])

    val_dataset = make_dataset(
        filelist=train_filelist,
        cropsize=cropsize*2,
        sr=44100,
        hop_length=hop_length,
        n_fft=fft,
        root=dir[1])

make_vocal_stems(dataset=vocal_dataset, root="G://", cropsize=cropsize*2, hop_length=hop_length, n_fft=fft)