# DiffuFont

DiffuFont is a RGB glyph diffusion project for Chinese font generation. The current training path is a pixel-space DiT x-pred model conditioned by a content glyph and multiple style reference glyphs.

The active path is:

1. Build the GB2312 common character set across all configured fonts.
2. Render RGB `128x128` glyph PNGs.
3. Pack content and training glyphs into LMDB.
4. Train `SourcePartRefDiT` with a content encoder, a style encoder, external content-style cross attention, and a DiT backbone.

Only the current RGB DiT training and inference path is documented here.

## Current Data

The rebuilt dataset uses:

- `CharacterData/CharList.json`: final common GB2312 character set.
- `DataPreparation/ContentFontList.json`: content font list.
- `DataPreparation/FontList.json`: style/target font list.
- `DataPreparation/LMDB/ContentFont.lmdb`: rendered content glyphs.
- `DataPreparation/LMDB/TrainFont.lmdb`: rendered target/style glyphs.

Images are stored and read as RGB tensors:

```text
glyph image: (3, 128, 128)
value range after transform: [-1, 1]
```

Rebuild the dataset:

```bash
conda run -n sg3 python fontprocessing/rebuild_gb2312_dataset.py \
  --project-root /scratch/yangximing/code/ass5-exp6/DiffuFont \
  --num-workers 48
```

## Training

Recommended launcher:

```bash
bash scripts/run_diffusion_colab.sh --foreground --device cuda:1
```

Default launcher settings:

```text
image_size = 128
patch_size = 8
encoder_hidden_dim = 256
dit_hidden_dim = 256
dit_depth = 12
dit_heads = 8
style_ref_count_min = 4
style_ref_count_max = 8
train_sampling = cartesian_font_char
cartesian_fonts_per_batch = 64
cartesian_chars_per_batch = 6
train_ratio = 0.95
```

With the current rebuilt dataset and `train_ratio=0.95`, the split is:

```text
train fonts = 301
train chars = 6424
train samples = 1,933,624
val fonts = 16
val chars = 339
val samples = 5,424
font overlap = 0
char overlap = 0
```

Validation evaluates unseen fonts and unseen characters. The fixed validation loader uses the first `16` validation fonts and first `16` validation characters as a cartesian grid.

Sample images saved during training use `16` examples:

```text
8 train font/char diagonal pairs
8 val font/char diagonal pairs
```

## Model

The active model is `SourcePartRefDiT`.

Content and style encoders share the same architecture but do not share weights:

```text
Input RGB:                  (B, 3, 128, 128)
ResDownBlock 3 -> 64:       (B, 64, 64, 64)
ResDownBlock 64 -> 128:     (B, 128, 32, 32)
ResDownBlock 128 -> 256:    (B, 256, 16, 16)
GN + SiLU + 1x1 Conv:       (B, 256, 16, 16)
flatten tokens:             (B, 256, 256)
```

The style bank shape is variable by batch:

```text
style_img:                  (U_style, R, 3, 128, 128)
style_tokens:               (U_style, R, 256, 256)
style_key/value:            (U_style, 1, 4, R*256, 64)
```

`R` is fixed inside one batch and sampled between `style_ref_count_min` and `style_ref_count_max` during training. All style references in a batch are valid, so cross attention runs as unmasked SDPA.

Model-internal tensor flow is in [docs/tensor_shapes.md](docs/tensor_shapes.md).

## Files

- `dataset.py`: LMDB dataset, font/char split, cartesian sampler.
- `train.py`: collators, fixed validation/sample batches, training entry.
- `models/source_part_ref_dit.py`: encoders, content-style cross attention, DiT wrapper.
- `models/diffusion_transformer_backbone.py`: pixel-space DiT backbone.
- `models/model.py`: trainer, x-pred loss, sampling and checkpointing.
- `DataPreparation/generate_font_images.py`: RGB glyph rendering.
- `fontprocessing/rebuild_gb2312_dataset.py`: GB2312 common charset and dataset rebuild.
