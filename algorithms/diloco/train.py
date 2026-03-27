"""
DiLoCo Inner Optimization — NanoGPT on Shakespeare
====================================================
Implements Algorithm 5 (Inner_Optimization) from the FEDSTR paper.

Inner optimizer: AdamW
Model: GPT (nanoGPT architecture, Karpathy 2022)
Dataset: Tiny Shakespeare (character-level)
"""

import os
import sys
import math
import pickle
import hashlib
import base64
import io
import json
from datetime import datetime
from typing import Dict, Optional

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPT, GPTConfig


def get_data_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "../../"))
    return os.path.join(repo_root, "data", "shakespeare")


def load_shard(data_dir, split, start_frac, end_frac):
    path = os.path.join(data_dir, f"{split}.bin")
    data = np.memmap(path, dtype=np.uint16, mode="r")
    total = len(data)
    s = int(start_frac * total)
    e = int(end_frac * total)
    return data[s:e]


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def load_meta(data_dir):
    with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
        return pickle.load(f)


def setup_log(dvm_id, round_num):
    log_dir = "/opt/fedstr/logs"
    os.makedirs(log_dir, exist_ok=True)
    os.chmod(log_dir, 0o777)
    log_path = os.path.join(log_dir, f"dvm_{dvm_id}_training.log")
    
    mode = "a" if os.path.exists(log_path) else "w"
    
    f = open(log_path, mode, encoding='utf-8', buffering=1)
    if mode == "w":
        f.write(f"FEDSTR DiLoCo Training Log - DVM {dvm_id}\n")
        f.write(f"Run started: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
        f.write("=" * 60 + "\n\n")
        f.flush()
    return f


def train_model(
    dataset_name,
    start_idx,
    end_idx,
    epochs,
    batch_size,
    initial_params=None,
    round_num=0,
    n_layer=4,
    n_head=4,
    n_embd=128,
    block_size=64,
    dropout=0.0,
    lr=1e-3,
    weight_decay=0.1,
    grad_clip=1.0,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if "cuda" in device else "cpu"

    # Get DVM identity and round from env
    dvm_id = os.environ.get("DVM_ID", "0")
    # round_num comes from job request via training.rs kwargs
    #round_num = int(os.environ.get("CURRENT_ROUND", "0"))

    print(f"  DiLoCo inner optimization — DVM {dvm_id}, Round {round_num}, device={device}")

    data_dir = get_data_dir()
    if not os.path.exists(os.path.join(data_dir, "train.bin")):
        print("  Dataset not found — running prepare_data.py...")
        from prepare_data import prepare
        prepare()

    meta = load_meta(data_dir)
    vocab_size = meta["vocab_size"]

    start_frac = start_idx / 60000.0
    end_frac = end_idx / 60000.0
    train_data = load_shard(data_dir, "train", start_frac, end_frac)
    val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    training_samples = len(train_data)

    config = GPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        bias=False,
    )
    model = GPT(config).to(device)

    if initial_params is not None:
        params_bytes = base64.b64decode(initial_params)
        state_dict = torch.load(io.BytesIO(params_bytes), weights_only=True, map_location=device)
        model.load_state_dict(state_dict)

    optimizer = model.configure_optimizers(
        weight_decay=weight_decay,
        learning_rate=lr,
        betas=(0.9, 0.95),
        device_type=device_type,
    )

    warmup_steps = min(100, max(1, epochs // 10))

    def get_lr(step):
        if step < warmup_steps:
            return lr * (step + 1) / warmup_steps
        if step > epochs:
            return lr * 0.1
        decay_ratio = (step - warmup_steps) / max(1, epochs - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return lr * 0.1 + coeff * (lr - lr * 0.1)

    # ── Open log file ────────────────────────────────────────────────────────
    log_file = setup_log(dvm_id, round_num)
    log_file.write(f"Round {round_num} - {epochs} inner steps, shard {start_frac:.0%}–{end_frac:.0%}\n")
    log_file.write(f"  Model: {config.n_layer}L {config.n_head}H {config.n_embd}d, vocab={vocab_size}\n")
    log_file.write(f"  Optimizer: AdamW lr={lr}, wd={weight_decay}, batch={batch_size}, block={block_size}\n")
    log_file.write(f"  From checkpoint: {initial_params is not None}\n\n")
    log_file.write(f"  {'Step':>6}  {'Train Loss':>10}  {'LR':>10}\n")
    log_file.write(f"  {'-'*6}  {'-'*10}  {'-'*10}\n")

    model.train()
    loss_history = []

    for step in range(epochs):
        current_lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        x, y = get_batch(train_data, block_size, batch_size, device)
        logits, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        # Log every step to file, print every 10% to console
        log_file.write(f"  {step+1:>6}  {loss_val:>10.4f}  {current_lr:>10.2e}\n")
        if step % max(1, epochs // 10) == 0 or step == epochs - 1:
            print(f"    step {step+1:4d}/{epochs} — loss: {loss_val:.4f}, lr: {current_lr:.2e}")

    # ── Validation ───────────────────────────────────────────────────────────
    model.eval()
    val_loss_sum = 0.0
    val_steps = 20
    with torch.no_grad():
        for _ in range(val_steps):
            x, y = get_batch(val_data, block_size, batch_size, device)
            _, loss = model(x, y)
            val_loss_sum += loss.item()
    val_loss = val_loss_sum / val_steps
    val_perplexity = math.exp(val_loss)
    final_loss = loss_history[-1]
    final_accuracy = 1.0 / val_perplexity * 100.0

    # ── Write round summary to log ───────────────────────────────────────────
    log_file.write(f"\n  Round {round_num} Summary:\n")
    log_file.write(f"    Final train loss:  {final_loss:.4f}\n")
    log_file.write(f"    Val loss:          {val_loss:.4f}\n")
    log_file.write(f"    Val perplexity:    {val_perplexity:.4f}\n")
    log_file.write(f"    Shard tokens:      {training_samples:,}\n")
    log_file.write("\n" + "-" * 60 + "\n\n")
    log_file.flush()
    log_file.close()

    print(f"\n  ✓ Round {round_num} complete")
    print(f"    Train loss:     {final_loss:.4f}")
    print(f"    Val loss:       {val_loss:.4f}")
    print(f"    Val perplexity: {val_perplexity:.4f}")

    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    model_bytes = buf.getvalue()
    model_hash = hashlib.sha256(model_bytes).hexdigest()
    model_base64 = base64.b64encode(model_bytes).decode("utf-8")

    return {
        "model_base64": model_base64,
        "model_hash": model_hash,
        "size_bytes": len(model_bytes),
        "final_loss": final_loss,
        "final_accuracy": final_accuracy,
        "loss_history": loss_history,
        "accuracy_history": [1.0 / math.exp(l) * 100.0 for l in loss_history],
        "epochs_completed": epochs,
        "training_samples": training_samples,
        "val_loss": val_loss,
        "val_perplexity": val_perplexity,
    }