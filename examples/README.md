# Examples

This directory keeps small, text-only examples that are safe to commit.

The runnable demo dataset is generated locally:

```powershell
python scripts/create_demo_data.py
```

That command writes demo images and `manifest.jsonl` under `data/raw/`. Files under `data/` are treated as local generated artifacts and are not tracked by Git.
