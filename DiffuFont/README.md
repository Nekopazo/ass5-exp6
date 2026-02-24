# DiffuFont (Current Branch)

This branch uses a source-aligned FontDiffuser backbone with configurable conditioning:

- `baseline`: no parts_vector, no RSI
- `parts_vector_only`: parts_vector on, RSI off
- `rsi_only`: parts_vector off, RSI on
- `full` (default): parts_vector on, RSI on

The old style-image `E_s(x_s)` conditioning path has been removed from runtime conditioning.

## Key Files

- `train.py`: main training entry
- `dataset.py`: dataset and PartBank retrieval
- `models/source_part_ref_unet.py`: parts_vector/RSI conditioning wrapper
- `models/source_fontdiffuser/`: source UNet/RSI blocks
- `models/model.py`: trainer, logging, sampling, checkpoints
- `docs/model_architecture.md`: latest architecture notes
- `docs/model_full_graph.md`: full graph diagram + layer-by-layer injection table
- `docs/scripts_usage.md`: all scripts, defaults, and usage

## Environment

Example (conda):

```bash
conda create -n sg3 python=3.10 -y
conda activate sg3
pip install torch torchvision torchaudio
pip install diffusers transformers lmdb pillow opencv-python
```

## Data Layout

```text
DiffuFont/
├── CharacterData/
│   ├── CharList.json
│   └── ReferenceCharList.json
├── DataPreparation/
│   ├── FontList.json
│   ├── LMDB/
│   │   ├── ContentFont.lmdb
│   │   └── TrainFont.lmdb
│   └── PartBank/
│       └── manifest.json
└── ...
```

## Training

Full conditioning:

```bash
python train.py \
  --data-root . \
  --conditioning-profile full \
  --part-retrieval-ep-ckpt checkpoints/e_p_font_encoder_best.pt \
  --save-dir checkpoints/run_full
```

Baseline:

```bash
python train.py \
  --data-root . \
  --conditioning-profile baseline \
  --save-dir checkpoints/run_baseline
```

Current default cadence:

- sample grid every `300` steps
- detailed log every `100` steps
- checkpoint every `5000` steps
- epoch checkpoint disabled by default

## Startup Config And Logs

At startup, `train.py` prints full run config and writes:

- `<save_dir>/train_run_config.json` (all args + derived runtime config)
- `<save_dir>/train_step_metrics.jsonl` (step-level metrics)

## Scripts

All script usage and defaults are documented in:

- `docs/scripts_usage.md`

Includes:

- `train.py`
- every `scripts/*.py`
- `scripts/run_train_a40_cuda1.sh`
- `scripts/run_train_a40_cuda2.sh`
