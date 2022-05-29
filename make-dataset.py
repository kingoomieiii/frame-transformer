from lib.dataset import make_validation_set, train_val_split, make_vocal_stems, make_dataset, make_mix_dataset

vocal_dataset = "G://vocals"
dataset = "G://new-inst"
validation = "E://Projects/vocal-remover-data/validation"

cropsize = 2048
hop_length = 1024
fft = 2048

dirs = [
    #("E://Projects/vocal-remover-data/training_p1", "C://Projects/vocal remover/vocal-remover-master/", True),
    #("E://Projects/vocal-remover-data/training_p2", "C://Projects/vocal remover/vocal-remover-master/", True),
    #("E://Projects/vocal-remover-data/training_p3", "", True),
    #("G://new", 'G://', False),
    #("C://new-inst", 'G://', True),
    #("G://training_p3", 'C://', True),
    ("G://full-albums", 'G://', True),
    #("G://training_p5", 'G://', False),
    #("E://Projects/vocal-remover-data/training_p3", 'C://', True),
    #("E://Projects/vocal-remover-data/validation", 'C://', False),
    #("G://new-inst", 'G://', True),
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

    val_dataset = make_mix_dataset(
        filelist=train_filelist,
        cropsize=cropsize,
        sr=44100,
        hop_length=hop_length,
        n_fft=fft,
        root=dir[1],
        is_validation=True)

# val_filelist, _ = train_val_split(
#     dataset_dir=validation,
#     val_filelist=[],
#     val_size=-1,
#     train_size=-1,
#     voxaug=False)

# val_dataset = make_dataset(
#     filelist=val_filelist,
#     cropsize=cropsize//2,
#     sr=44100,
#     hop_length=hop_length,
#     n_fft=fft,
#     root="D://",
#     is_validation=True)

#make_vocal_stems(dataset=vocal_dataset, root="G://", cropsize=cropsize, hop_length=hop_length, n_fft=fft)