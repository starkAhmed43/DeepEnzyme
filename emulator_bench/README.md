# DeepEnzyme EMULaToR bench

This bench retrains DeepEnzyme on explicit EMULaToR train/val/test splits while keeping the original DeepEnzyme model settings as the default path.

Default data root:

```bash
/home/adhil/github/EMULaToR/data/processed/baselines/DeepEnzyme
```

Default GPU selection is external:

```bash
CUDA_VISIBLE_DEVICES=2
```

Inside Python, scripts use `cuda:0`, which maps to physical GPU 2 when the environment variable above is set.

## Inputs and features

Per sample, the bench uses:

- `sequence`: protein amino-acid sequence.
- `smiles`: substrate SMILES.
- `target`: prepared from `log10_value` by default for EMULaToR comparability.
- `structure_path` and `chain_id`: one resolved protein structure per row.

Static DeepEnzyme inputs are cached once under:

```bash
/home/adhil/github/EMULaToR/data/processed/baselines/DeepEnzyme/embeddings
```

Cached artifacts include:

- sequence 4-gram token IDs,
- substrate Weisfeiler-Lehman fingerprint IDs,
- substrate adjacency matrices,
- protein contact maps from C-alpha distances under the original 10A threshold.

These are static inputs to DeepEnzyme's trainable embedding layers. The neural embedding outputs themselves are not cached because they change during retraining.

## Structure resolution

`prepare_splits.py` resolves one structure per row:

1. select the best aligned experimental PDB from `/home/adhil/github/EMULaToR/data/intermediate/processed_exp_pdb`,
2. fall back to AlphaFold structures from `/home/adhil/github/EMULaToR/data/intermediate/alphafold`,
3. fall back to ESM structures from `/home/adhil/github/EMULaToR/data/intermediate/esm`.

PDB sequence parsing and selected structure decisions are cached in:

```bash
/home/adhil/github/EMULaToR/data/processed/baselines/DeepEnzyme/_structure_alignment_cache
```

## Original-settings retraining

The default trainer uses the original settings:

- `lr=0.001`
- `iteration=200`
- `weight_decay=1e-6`
- `dropout=0.3`
- `dim=64`
- `layer_output=3`
- `hidden_dim1=64`
- `hidden_dim2=64`
- `nhead=4`
- `hid_size=64`
- `layers_trans=3`

It preserves the original sample-wise training loop and StepLR stepping behavior, but uses explicit EMULaToR train/val/test splits.

The bench-local model implementation removes hardcoded `.cuda()` calls so `CUDA_VISIBLE_DEVICES=2` works reliably.

## AMP and resume

On CUDA, the trainer enables TF32 and automatic mixed precision:

- bf16 on Ampere-or-newer GPUs,
- fp16 otherwise,
- fp32 on CPU or with `--no_amp`.

Each run writes:

- `last_checkpoint.pt`
- `best_checkpoint.pt`
- `logfile.csv`
- `final_results_train.csv`
- `final_results_val.csv`
- `final_results_test.csv`
- `pred_label_train.csv`
- `pred_label_val.csv`
- `pred_label_test.csv`
- `completed.json`

Restarting the same command resumes from `last_checkpoint.pt`. Completed runs are skipped unless `--overwrite_runs` is passed to the runner or `--overwrite` is passed to the trainer.

## Optional Optuna

Optuna is included but is not part of the default retraining path. It tunes only retraining-safe knobs:

- learning rate,
- weight decay,
- dropout.

The model architecture and feature definitions remain fixed at the original DeepEnzyme defaults.

See `commands.txt` for exact commands.

