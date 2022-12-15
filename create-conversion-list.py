import argparse
import json
import os

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=str, default=None)
    args = p.parse_args()

    name = os.path.split(args.input)[-1]
    extensions = ['wav', 'm4a', 'mp3', 'mp4', 'flac']

    files = []
    d = os.listdir(args.input)
    for f in d:
        ext = f[::-1].split('.')[0][::-1]

        if ext in extensions:
            files.append(f)

    data = {
        "output": name,
        "convert": [
            f'{args.input}/{f}' for f in files
        ]
    }

    obj = json.dumps(data, indent=4)

    with open(f'convert/{name}.json', "w") as out:
        out.write(obj)

if __name__ == '__main__':
    main()
