from __future__ import annotations

import random

import pandas as pd


def assign_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    result = df.copy()
    result["split"] = "excluded"
    accepted_indices = result.index[result["filter_status"] == "accepted"].tolist()
    rng = random.Random(seed)
    rng.shuffle(accepted_indices)

    total = len(accepted_indices)
    train_cutoff = int(total * train_ratio)
    val_cutoff = train_cutoff + int(total * val_ratio)

    for pos, idx in enumerate(accepted_indices):
        if pos < train_cutoff:
            result.at[idx, "split"] = "train"
        elif pos < val_cutoff:
            result.at[idx, "split"] = "val"
        else:
            result.at[idx, "split"] = "eval"
    return result
