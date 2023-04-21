import os
import numpy as np
from tqdm import tqdm

files = os.listdir('./')

base_names = {}
for f in files:
    if not f.endswith('.npz'):
        continue

    fi = f[::-1]

    idx = fi.index('p_') + 2
    fi2 = fi[idx:][::-1]

    if base_names.get(fi2) is None:
        base_names[fi2] = []

    base_names[fi2].append(f)

for key in tqdm(base_names.keys()):
    files = base_names[key]

    ci, cr = 0, 0
    datas = []

    for file in files:
        data = np.load(file)
        X = data['X']
        cr = np.max([cr, np.abs(X.real).max()])
        ci = np.max([ci, np.abs(X.imag).max()])
        datas.append((file, data))

    for file, data in datas:
        np.savez(file, cr=cr, ci=ci, **data)