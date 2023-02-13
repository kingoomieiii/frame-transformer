from lib.dataset import make_dataset_translate, make_validation_set, train_val_split, make_vocal_stems, make_dataset, make_mix_dataset, make_vocal_stems2

vocal_dataset = "J://nvox"
validation = "G://dataset/validation"


vocal_datasets = [
    ("H://vox4", "D://"),
]

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
    # ("H://inst", 'H://', True, False),
    # ("J://dataset/inst-01", 'C://', True, False),
    # ("J://dataset/inst-01", 'C://', True, False),
    # ("J://dataset/inst-03", 'D://', True, False),
    ("H://inst", 'H://', True, False),
    # ("J://dataset/training_p1", 'D://', True, False),
    # ("J://dataset/training_p11", 'D://', True, False),
    # ("J://dataset/training_p12", 'D://', True, False),
    # ("K://dataset/training_p2", 'F://', True, False),
    # ("K://dataset/training_p3", 'F://', True, False),
    # ("K://dataset/training_p4", 'F://', True, False),
    # ("J://dataset/training_p12", 'H://', True, False),
    # ("K://dataset/training_p6", 'H://', True, False),
    # ("K://dataset/training_p7", 'H://', True, False),
    # ("J://dataset/training_p13", 'H://', True, False),
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
        voxaug=dir[2] and not dir[3])

    if not dir[2] or dir[3]:
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
            suffix='_PAIRS')
    else:
        val_dataset = make_dataset_translate(
            filelist=train_filelist,
            cropsize=cropsize,
            sr=44100,
            hop_length=hop_length,
            n_fft=fft,
            root=dir[1],
            is_validation=True,
            suffix='_DRUMS')

# val_filelist, _ = train_val_split(
#     dataset_dir=validation,
#     val_filelist=[],
#     val_size=-1,
#     train_size=-1,
#     voxaug=False)

# val_dataset = make_dataset(
#     filelist=val_filelist,
#     cropsize=cropsize,
#     sr=44100,
#     hop_length=hop_length, 
#     n_fft=fft,
#     root="C://",
#     is_validation=True,
#     suffix='_VALIDATION')

for dir in vocal_datasets:
    make_vocal_stems2(dataset=dir[0], root=dir[1], cropsize=cropsize, hop_length=hop_length, n_fft=fft)