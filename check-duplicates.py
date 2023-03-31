
import argparse
import os
import shutil

p = argparse.ArgumentParser()
p.add_argument('--input_dir', type=str, default=None)
args = p.parse_args()

output_dirs = [
    # '/home/ben/cs2048_sr44100_hl1024_nf2048_of0',
    # '/media/ben/internal-nvme-b/cs2048_sr44100_hl1024_nf2048_of0',
    '/home/ben/cs2048_sr44100_hl1024_nf2048_of0_VOCALS',
    '/media/ben/internal-nvme-b/cs2048_sr44100_hl1024_nf2048_of0_VOCALS',
    # '/media/ben/Evo 870 SATA 1/dataset/instruments/training_p1',
    # '/media/ben/Evo 870 SATA 1/dataset/instruments/training_p10',
    # '/media/ben/Evo 870 SATA 1/dataset/instruments/training_p2',
    # '/media/ben/Evo 870 SATA 1/dataset/instruments/training_p3',
    # '/media/ben/Evo 870 SATA 1/dataset/instruments/training_p11',
    # '/media/ben/Evo 870 SATA 1/dataset/instruments/training_p4',
    # '/media/ben/Evo 870 SATA 1/dataset/instruments/training_p15',
    # '/media/ben/Evo 870 SATA 1/dataset/instruments/training_p13',
    # '/media/ben/Evo 870 SATA 1/dataset/instruments/training_p5',
    # '/media/ben/Evo 870 SATA 1/dataset/instruments/training_p7',
    # '/media/ben/Evo 870 SATA 1/dataset/instruments/training_p12',
    # '/media/ben/Evo 870 SATA 1/dataset/instruments/training_p14',
    # '/media/ben/Evo 870 SATA 1/dataset/instruments/training_p6',
    # '/media/ben/Evo 870 SATA 1/dataset/instruments/training_p9',
    # '/media/ben/Evo 870 SATA 1/dataset/instruments/training_p16',
]

for dir1 in output_dirs:
    md = dir1
    df, files = os.listdir(md), []

    for file in df:
        if os.path.isfile(f'{md}/{file}'):
            files.append(file)

    for dir2 in output_dirs:
        if dir1 != dir2:
            for file in files:
                if os.path.exists(f'{os.path.join(dir2, "")}/{os.path.basename(file)}'):
                    if not os.path.exists(f'{os.path.join(dir2, "")}/dup'):
                        os.mkdir(f'{os.path.join(dir2, "")}/dup')

                    print(file)
                    shutil.move(f'{os.path.join(dir2, "")}/{os.path.basename(file)}', f'{os.path.join(dir2, "")}/dup/{os.path.basename(file)}')