# Few-Shot Font Generation via Denoising Diffusion and Component-Level Fine-Grained Style

## Key Components

* `models/font_diffusion_unet.py` – U-Net generator containing DACA / FGSA / AdaLN modules.
* `models/model.py` – training utilities (losses, DDPM training, DDIM sampling, automatic image logging).
* `train.py` – single-GPU training script.
* `dataset.py` – `FontImageDataset` loader used by the project.

## 1. Installation
```bash
# Python ≥ 3.8 is recommended
conda create -n DiffFont python=3.9 -y
conda activate DiffFont

# Core dependencies (choose the CUDA / CPU build that matches your system)
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
pip install tqdm lmdb pillow
pip install opencv-python
```
A CUDA-enabled GPU is strongly recommended for training.

## 2. Data Preparation
The directory layout expected by `dataset.py` is shown below.
```text
ProjectRoot/
├── CharacterData/
│   ├── CharList.json          # list of content characters
│   └── ReferenceCharList.json # global style reference pool (e.g. 300 chars)
├── DataPreparation/
│   ├── FontList.json          # list of candidate .ttf files
│   └── LMDB/
│       ├── ContentFont.lmdb   # LMDB for the content font
│       └── TrainFont.lmdb     # LMDB for all style / target fonts
└── ...
```
### 2.1 Charset Preparation (No Mapping)

```bash
python scripts/prepare_common_charset.py \
  --out-dir CharacterData \
  --char-count 2000 \
  --ref-count 300
```

Outputs:
- `CharacterData/CharList.json`
- `CharacterData/ReferenceCharList.json`
- `CharacterData/dataset_meta.json`

No `mapping.json` is generated.

### 2.2 Image Generation & LMDB Construction
Two helper scripts are provided:
* **`generate_font_images.py`** – render characters of a given TrueType font to individual PNG files.
* **`images_to_lmdb.py`** – convert a folder of PNGs into an LMDB database.

Typical workflow
1. **Generate content-font images**
   ```bash
   # Render one font by index from DataPreparation/FontList.json
   python DataPreparation/generate_font_images.py \
       --project-root . \
       --font-indices 0 \
       --char-list-json CharacterData/CharList.json \
       --out-dir DataPreparation/Generated/ContentFont

   # Convert to LMDB
   python DataPreparation/images_to_lmdb.py \
       --project-root . \
       --img-roots DataPreparation/Generated/ContentFont \
       --lmdb-path DataPreparation/LMDB/ContentFont.lmdb
   ```
2. **Generate style / target font images**
   ```bash
   # Render all fonts listed in FontList.json
   python DataPreparation/generate_font_images.py \
       --project-root . \
       --char-list-json CharacterData/CharList.json \
       --out-dir DataPreparation/Generated/TrainFonts

   # Convert to a single LMDB (sub-directories are supported)
   python DataPreparation/images_to_lmdb.py \
       --project-root . \
       --img-roots DataPreparation/Generated/TrainFonts \
       --lmdb-path DataPreparation/LMDB/TrainFont.lmdb
   ```
> The default filename pattern is `FontName@<char>.png`; `images_to_lmdb.py` relies on this convention.

### 2.3 Font Files
The **300** Chinese font files used in our experiments can be downloaded from Google Drive:
<https://drive.google.com/file/d/17rajeJz53RnCOEv9B4X6tDKaIwpz2bIb/view?usp=drive_link>

After downloading, extract the archive to `DataPreparation/Fonts/` and list the extracted `.ttf` paths in `DataPreparation/FontList.json`, then follow Section 2.1 to render the images.

### 2.4 Style Reference Pool

`ReferenceCharList.json` contains a shared pool of style reference chars.
At training time, `dataset.py` samples `num_style_refs` style chars from this pool
(default: same as `--style-k`), so no per-content mapping file is required.

`FontImageDataset` will first try the font from `FontList.json` + `font_index`.
If that font does not exist in `TrainFont.lmdb`, it automatically falls back to a usable
font prefix found in LMDB keys (format: `<FontName>@<char>`).

### 2.5 Build Few-Part Bank (Optional but Recommended)

The script below extracts representative local patches (few-parts) for each font.
It uses keypoints/descriptors (SIFT if available, ORB fallback) and descriptor-space
diversity sampling.

```bash
python scripts/build_part_bank.py \
  --project-root . \
  --fonts-dir fonts \
  --charset-json CharacterData/ReferenceCharList.json \
  --output-dir DataPreparation/PartBank \
  --parts-per-font 32 \
  --patch-size 64 \
  --detector sift \
  --canvas-size 256 \
  --char-size 224
```

Output:
- `DataPreparation/PartBank/<FontName>/part_*.png`
- `DataPreparation/PartBank/manifest.json`

### 2.6 Pretrain Part-Style Encoder (Optional)

```bash
python scripts/pretrain_part_style_encoder.py \
  --project-root . \
  --manifest DataPreparation/PartBank/manifest.json \
  --out checkpoints/part_style_encoder_pretrain_final.pt \
  --best-out checkpoints/part_style_encoder_pretrain_best.pt \
  --log-file checkpoints/part_style_encoder_pretrain.log \
  --metrics-jsonl checkpoints/part_style_encoder_pretrain.metrics.jsonl \
  --steps 10000 \
  --batch-size 64 \
  --min-set-size 1 \
  --warmup-max-set-size 6 \
  --warmup-steps 4000 \
  --max-set-size 12 \
  --val-ratio 0.1 \
  --val-batches 8 \
  --monitor val_loss \
  --early-stop-patience 20 \
  --patch-size 64 \
  --temperature 0.4 \
  --log-every 100
```

### 2.7 Analyze Component-Overlap Sampling (Recommended)

```bash
python scripts/analyze_component_overlap.py \
  --project-root . \
  --font-mode random \
  --style-k 3 \
  --samples 4000 \
  --out-dir checkpoints/overlap_stats
```

Output:
- `checkpoints/overlap_stats/component_overlap_report.json`
- `checkpoints/overlap_stats/component_overlap_top_pairs.csv`
- `checkpoints/overlap_stats/component_overlap_hist.png` (if matplotlib is available)

## 3. Training
```bash
python train.py \
  --data-root . \
  --epochs 50 \
  --batch 1 \
  --font-mode random \
  --component-guided-style \
  --style-k 3 \
  --daca-layers 0,1,1,0 \
  --fgsa-layers 1,1,1,0 \
  --attnx-enabled \
  --use-part-style \
  --part-patch-size 64 \
  --part-patch-stride 32 \
  --part-min-patches-per-style 2 \
  --part-max-patches-per-style 8 \
  --part-style-pretrained checkpoints/part_style_encoder_pretrain_best.pt \
  --overlap-report-samples 2000 \
  --overlap-report-json checkpoints/overlap_stats/train_overlap_report.json \
  --lr 2e-4 \
  --save-dir checkpoints
```

If you want to keep the pretrained part encoder fixed:

```bash
python train.py ... --use-part-style --part-style-pretrained checkpoints/part_style_encoder_pretrain_best.pt --freeze-part-style
```

During training
* A checkpoint is saved every **5** epochs in `checkpoints/ckpt_*.pt`.
* A DDIM sample is saved every `--sample-every-steps` mini-batches (default **100**) in `checkpoints/samples/` with filename `sample_ep{epoch}_step{batch}.png`.

Resume training
```bash
python train.py --resume checkpoints/ckpt_10.pt
```

## 4. Inference / Sampling
```python
import torch
from PIL import Image
from torchvision import transforms as T
from models.font_diffusion_unet import FontDiffusionUNet
from models.model import DiffusionTrainer

net = FontDiffusionUNet(in_channels=3, style_k=3)
trainer = DiffusionTrainer(net, torch.device('cuda'), sample_every_steps=None)
trainer.load('checkpoints/ckpt_50.pt')

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(0.5, 0.5),
])
content_img = transform(Image.open('content.png')).unsqueeze(0)
style_imgs = [transform(Image.open(f'style_{i}.png')) for i in range(3)]
style_img = torch.cat(style_imgs, dim=0).unsqueeze(0)

sample = trainer.ddim_sample(content_img, style_img, c=10, eta=0)
T.functional.to_pil_image((sample[0] + 1) / 2).save('result.png')
```

---
