
import argparse

from libft2gan.dataset_voxaug import VoxAugDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--instrumental_lib', type=str, default="C://cs2048_sr44100_hl1024_nf2048_of0|D://cs2048_sr44100_hl1024_nf2048_of0|F://cs2048_sr44100_hl1024_nf2048_of0|H://cs2048_sr44100_hl1024_nf2048_of0")
    p.add_argument('--vocal_lib', type=str, default="C://cs2048_sr44100_hl1024_nf2048_of0_VOCALS|D://cs2048_sr44100_hl1024_nf2048_of0_VOCALS")
    p.add_argument('--validation_lib', type=str, default="C://cs2048_sr44100_hl1024_nf2048_of0_VALIDATION")
    # p.add_argument('--instrumental_lib', type=str, default="/home/ben/cs2048_sr44100_hl1024_nf2048_of0|/media/ben/internal-nvme-b/cs2048_sr44100_hl1024_nf2048_of0")
    # p.add_argument('--vocal_lib', type=str, default="/home/ben/cs2048_sr44100_hl1024_nf2048_of0_VOCALS")
    # p.add_argument('--validation_lib', type=str, default="/media/ben/internal-nvme-b/cs2048_sr44100_hl1024_nf2048_of0_VALIDATION")

    args = p.parse_args()
    args.instrumental_lib = [p for p in args.instrumental_lib.split('|')]
    args.vocal_lib = [p for p in args.vocal_lib.split('|')]

    train_dataset = VoxAugDataset(
        path=args.instrumental_lib,
        vocal_path=args.vocal_lib,
        is_validation=False
    )

    val_dataset = VoxAugDataset(
        path=[args.validation_lib],
        vocal_path=None,
        is_validation=True
    )

    print(f'mix count: {len(train_dataset.curr_list)}')
    print(f'vocal count: {len(train_dataset.vocal_list)}')
    print(f'val count: {len(val_dataset.curr_list)}')    

if __name__ == '__main__':
    main()