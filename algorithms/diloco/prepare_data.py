"""
Shakespeare Dataset Preparation
================================
Downloads tiny Shakespeare dataset and creates train/val splits
as binary files for fast loading during DiLoCo training.

Run once before training:
    python algorithms/diloco/prepare_data.py

Creates:
    data/shakespeare/train.bin  - training tokens
    data/shakespeare/val.bin    - validation tokens
    data/shakespeare/meta.pkl   - vocab info (chars, stoi, itos)
"""

import os
import pickle
import requests
import numpy as np


def prepare():
    data_dir = os.path.join(os.path.dirname(__file__), '../../data/shakespeare')
    os.makedirs(data_dir, exist_ok=True)

    input_file = os.path.join(data_dir, 'input.txt')

    if not os.path.exists(input_file):
        print("Downloading Shakespeare dataset...")
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        r = requests.get(url)
        with open(input_file, 'w') as f:
            f.write(r.text)
        print(f"  Downloaded {len(r.text):,} characters")
    else:
        with open(input_file, 'r') as f:
            data = f.read()
        print(f"  Found existing dataset: {len(data):,} characters")

    with open(input_file, 'r') as f:
        data = f.read()

    # Build character vocabulary
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print(f"  Vocab size: {vocab_size} unique characters")

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]

    # Train/val split: 90/10
    n = len(data)
    train_data = data[:int(n * 0.9)]
    val_data = data[int(n * 0.9):]

    train_ids = np.array(encode(train_data), dtype=np.uint16)
    val_ids = np.array(encode(val_data), dtype=np.uint16)

    print(f"  Train tokens: {len(train_ids):,}")
    print(f"  Val tokens:   {len(val_ids):,}")

    train_ids.tofile(os.path.join(data_dir, 'train.bin'))
    val_ids.tofile(os.path.join(data_dir, 'val.bin'))

    meta = {'vocab_size': vocab_size, 'itos': itos, 'stoi': stoi}
    with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    print(f"  Saved to {data_dir}/")
    print(f"  ✓ Dataset ready. vocab_size={vocab_size}")
    return vocab_size


if __name__ == '__main__':
    prepare()
