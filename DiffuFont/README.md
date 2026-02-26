# DiffuFont (Current Branch)

This branch now follows the "few-part font generation" redesign:

- strict UNet resolution path: `256 -> 128(stem) -> 64 -> 32 -> 16 -> 32 -> 64 -> 128 -> 256(head)`
- style conditioning from online retrieved PartBank parts only
- DeepSets `mean + LayerNorm` set aggregation, then MLP to fixed style tokens (`M=8`, `d=256`)
- Mid block content injection disabled by design
- RSI branch kept and controlled by conditioning profile (default off)

## Conditioning Profiles

- `baseline`: no parts_vector, no RSI
- `parts_vector_only` (default): parts_vector on, RSI off
- `rsi_only`: parts_vector off, RSI on
- `full`: parts_vector on, RSI on

## Key Files

- `train.py`: main training entry
- `dataset.py`: dataset and online PartBank retrieval
- `models/source_part_ref_unet.py`: stem/head, parts-token encoder, wrapper
- `models/source_fontdiffuser/unet.py`: UNet topology and scale gating
- `models/source_fontdiffuser/unet_blocks.py`: MCA/RSI blocks and mid content-attn switch
- `models/model.py`: trainer, sampling, checkpoints
- `docs/model_architecture.md`: latest architecture notes
- `docs/model_full_graph.md`: full graph + layer injection table
- `docs/scripts_usage.md`: scripts and defaults

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
│   │   ├── TrainFont.lmdb
│   │   └── PartBank.lmdb
│   └── PartBank/
│       └── manifest.json
└── ...
```

## Training

Default (parts tokens on, RSI off):

```bash
python train.py \
  --data-root . \
  --conditioning-profile parts_vector_only \
  --attn-scales 16,32 \
  --part-retrieval-ep-ckpt checkpoints/e_p_font_encoder_best.pt \
  --part-set-size 32 \
  --part-set-min-size 32 \
  --style-token-count 8 \
  --style-token-dim 256 \
  --save-dir checkpoints/run_parts_only
```

Enable RSI:

```bash
python train.py \
  --data-root . \
  --conditioning-profile full \
  --attn-scales 16,32 \
  --part-retrieval-ep-ckpt checkpoints/e_p_font_encoder_best.pt \
  --save-dir checkpoints/run_full
```

## Logging / Checkpoints

- sample grid every `300` steps
- detailed log every `100` steps
- checkpoint every `5000` steps
- epoch checkpoint disabled by default

`train.py` writes:

- `<save_dir>/train_run_config.json`
- `<save_dir>/train_step_metrics.jsonl`
