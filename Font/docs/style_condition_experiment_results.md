# Style Conditioning Experiment Results

## Evaluation Setup

All results below use the same evaluation protocol.

| Item | Value |
|---|---|
| Evaluation split | `fixed_full_val` |
| Checkpoint | `latest.pt` |
| Style references | `3` |
| Style reference selection | first 3 images from the candidate set |
| Inference steps | `20` |
| Eval fonts per batch | `8` |
| Eval chars per batch | `48` |
| Samples | `5424` |
| Training step | `100000` |

## Configuration Definitions

| Configuration | Content encoder | Style encoder | Style conditioning |
|---|---|---|---|
| `cnn_tokenwise_cross` | CNN encoder | CNN encoder | Style tokens are kept as tokenwise features and fused with content tokens by cross attention. |
| `cnn_global_mean` | CNN encoder | CNN encoder | Style token features are averaged into one global style vector. |
| `cnn_global_cls` | CNN encoder | CNN encoder | CNN produces final feature tokens; one learnable CLS query performs attention pooling once at the final style feature layer to produce a global style vector. |
| `vit_tokenwise_cross` | ViT encoder | ViT encoder | Style tokens are kept as tokenwise features and fused with content tokens by cross attention. |
| `vit_global_mean` | ViT encoder | ViT encoder | Style token features are averaged into one global style vector. |
| `vit_global_cls` | ViT encoder | ViT encoder | Style encoder uses a CLS token through all ViT layers; the final CLS output is used as the global style vector. |
| `cnn_cross_mlp_residual` | CNN encoder | CNN encoder | Style tokens are fused by cross attention; the fused tokens are then updated with `cross + FFN(cross)`. |
| `cnn_cross_mlp_x` | CNN encoder | CNN encoder | Style tokens are fused by cross attention; the fused tokens are replaced by `FFN(cross)`. Training prediction target is `x`. |
| `cnn_cross_noise` | CNN encoder | CNN encoder | Style tokens are fused by cross attention only. Training prediction target is `noise`. |
| `cnn_cross_velocity` | CNN encoder | CNN encoder | Style tokens are fused by cross attention only. Training prediction target is `velocity`. |
| `cnn_cross_x` | CNN encoder | CNN encoder | Style tokens are fused by cross attention only. Training prediction target is `x`; training uses dynamic `1-3` style references. |
| `cnn_cross_x_perc0002` | CNN encoder | CNN encoder | Style tokens are fused by cross attention only. Training prediction target is `x`; perceptual loss coefficient is `0.002`. |
| `lite_dwres_cnn_cross_x_ref3` | Lite-DWRes CNN encoder | Lite-DWRes CNN encoder | Style tokens are fused by cross attention only. Training prediction target is `x`; encoder variant is `lite_dwres`, with fixed `3` style references. |
| `full_conv_stage_res_cnn_cross_x_ref3` | CNN residual-refinement encoder | CNN residual-refinement encoder | Style tokens are fused by cross attention only. Training prediction target is `x`; encoder variant is `full_conv_stage_res`, with fixed `3` style references. |
| `lite_dwres_all_cnn_cross_x_ref3` | Lite-DWRes-All CNN encoder | Lite-DWRes-All CNN encoder | Style tokens are fused by cross attention only. Training prediction target is `x`; encoder variant is `lite_dwres_all`, with fixed `3` style references. |

## Architecture Notes

| Encoder | Structure |
|---|---|
| CNN | Original CNN encoder from the previous git version: `3 -> 32` stem, then three downsample-first convolution blocks: `32 -> 64`, `64 -> 128`, `128 -> 256`. |
| ViT | Four-layer ViT encoder, hidden dimension `256`, attention heads `4`. |
| Cross attention | `4` heads for content-style fusion. |

## Parameter Counts

Parameter counts are reported for the active architecture only. Unused CLS pooling parameters that were accidentally registered in some historical `tokenwise_cross` CNN checkpoints are excluded.

| Configuration | Parameters |
|---|---:|
| `cnn_tokenwise_cross` | 46,523,984 |
| `cnn_global_mean` | 46,260,688 |
| `cnn_global_cls` | 47,050,404 |
| `vit_tokenwise_cross` | 50,609,968 |
| `vit_global_mean` | 50,346,672 |
| `vit_global_cls` | 50,346,928 |
| `cnn_cross_mlp_residual` | 47,049,636 |
| `cnn_cross_mlp_x` | 47,049,636 |
| `cnn_cross_noise` | 46,523,984 |
| `cnn_cross_velocity` | 46,523,984 |
| `cnn_cross_x` | 46,523,984 |
| `cnn_cross_x_perc0002` | 46,523,984 |
| `lite_dwres_cnn_cross_x_ref3` | 45,222,608 |
| `full_conv_stage_res_cnn_cross_x_ref3` | 46,523,984 |
| `lite_dwres_all_cnn_cross_x_ref3` | 45,158,480 |

## Full Results

| Configuration | Run directory | Parameters | FID ↓ | L1 ↓ | SSIM ↑ | LPIPS ↓ | Eval time (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cnn_tokenwise_cross` | `checkpoints/xpred_20260601_204907_cnn_tokenwise_cross` | 46,523,984 | 6.509067 | 0.151751 | 0.567950 | 0.174092 | 212.718 |
| `cnn_global_mean` | `checkpoints/xpred_20260601_204931_cnn_global_mean` | 46,260,688 | 7.111378 | 0.153166 | 0.565824 | 0.178171 | 214.364 |
| `cnn_global_cls` | `checkpoints/xpred_20260603_134809_cnn_global_cls` | 47,050,404 | 6.865901 | 0.157640 | 0.557014 | 0.179370 | 213.008 |
| `vit_tokenwise_cross` | `checkpoints/xpred_20260602_153201_vit_tokenwise_cross` | 50,609,968 | 7.527495 | 0.158107 | 0.555811 | 0.186114 | 213.161 |
| `vit_global_mean` | `checkpoints/xpred_20260602_153043_vit_global_mean` | 50,346,672 | 8.079024 | 0.159189 | 0.552365 | 0.184739 | 212.340 |
| `vit_global_cls` | `checkpoints/xpred_20260603_134831_vit_global_cls` | 50,346,928 | 7.854888 | 0.157753 | 0.555768 | 0.180119 | 343.683 |
| `cnn_cross_mlp_residual` | `checkpoints/xpred_20260604_135831_cnn + tokenwise_cross + cross_mlp_residual` | 47,049,636 | 6.788112 | 0.156760 | 0.559991 | 0.175019 | 213.523 |
| `cnn_cross_mlp_x` | `checkpoints/xpred_20260605_083238_cnn_tokenwise_cross_cross_mlp_x` | 47,049,636 | 7.308525 | 0.156038 | 0.561535 | 0.175982 | 212.970 |
| `cnn_cross_noise` | `checkpoints/xpred_20260605_084110_cnn_tokenwise_cross_cross_noise` | 46,523,984 | 50.160320 | 0.160104 | 0.549915 | 0.178915 | 214.381 |
| `cnn_cross_velocity` | `checkpoints/xpred_20260606_055508_cnn_tokenwise_cross_cross_velocity` | 46,523,984 | 44.305065 | 0.157483 | 0.555455 | 0.177398 | 212.922 |
| `cnn_cross_x` | `checkpoints/xpred_20260606_055630_cnn_tokenwise_cross_cross_x` | 46,523,984 | 7.119216 | 0.154684 | 0.562997 | 0.178193 | 214.505 |
| `cnn_cross_x_perc0002` | `checkpoints/xpred_20260607_191626_cnn_tokenwise_cross_cross_x_perc0002` | 46,523,984 | 6.428864 | 0.152091 | 0.568248 | 0.170205 | 308.587 |
| `lite_dwres_cnn_cross_x_ref3` | `checkpoints/xpred_20260609_100404_lite_dwres_cnn_tokenwise_cross_cross_x_ref3` | 45,222,608 | 5.969135 | 0.153371 | 0.565327 | 0.173842 | 340.093 |
| `full_conv_stage_res_cnn_cross_x_ref3` | `checkpoints/xpred_20260610_230306_full_conv_stage_res_cnn_tokenwise_cross_cross_x_ref3` | 46,523,984 | 6.118855 | 0.152598 | 0.566585 | 0.172398 | 416.220 |
| `lite_dwres_all_cnn_cross_x_ref3` | `checkpoints/xpred_20260614_001143_lite_dwres_all_cnn_tokenwise_cross_cross_x_ref3` | 45,158,480 | 6.043400 | 0.154137 | 0.563864 | 0.173993 | 380.428 |

## Inference Compute

FLOPs are normalized to the same inference protocol: one generated sample, `3` style references, `20` inference steps, and Heun sampling. A multiply-add is counted as `2` FLOPs. With Heun sampling, `20` inference steps call the denoiser `39` times. `Normalized compute` is relative to `cnn_cross_x`.

| Configuration | Params | Condition precompute GFLOPs | Per denoiser call GFLOPs | Denoiser calls | Total GFLOPs / sample | Total TFLOPs / sample | Normalized compute |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cnn_tokenwise_cross` | 46,523,984 | 7.028343 | 20.792486 | 39 | 817.935312 | 0.817935 | 1.000000 |
| `cnn_global_mean` | 46,260,688 | 5.956960 | 20.792486 | 39 | 816.863930 | 0.816864 | 0.998690 |
| `cnn_global_cls` | 47,050,404 | 6.163002 | 20.792486 | 39 | 817.069972 | 0.817070 | 0.998942 |
| `vit_tokenwise_cross` | 50,609,968 | 9.091940 | 20.792486 | 39 | 819.998910 | 0.819999 | 1.002523 |
| `vit_global_mean` | 50,346,672 | 8.020558 | 20.792486 | 39 | 818.927527 | 0.818928 | 1.001213 |
| `vit_global_cls` | 50,346,928 | 8.045724 | 20.792486 | 39 | 818.952693 | 0.818953 | 1.001244 |
| `cnn_cross_mlp_residual` | 47,049,636 | 7.296931 | 20.792486 | 39 | 818.203900 | 0.818204 | 1.000328 |
| `cnn_cross_mlp_x` | 47,049,636 | 7.296931 | 20.792486 | 39 | 818.203900 | 0.818204 | 1.000328 |
| `cnn_cross_noise` | 46,523,984 | 7.028343 | 20.792486 | 39 | 817.935312 | 0.817935 | 1.000000 |
| `cnn_cross_velocity` | 46,523,984 | 7.028343 | 20.792486 | 39 | 817.935312 | 0.817935 | 1.000000 |
| `cnn_cross_x` | 46,523,984 | 7.028343 | 20.792486 | 39 | 817.935312 | 0.817935 | 1.000000 |
| `cnn_cross_x_perc0002` | 46,523,984 | 7.028343 | 20.792486 | 39 | 817.935312 | 0.817935 | 1.000000 |
| `lite_dwres_cnn_cross_x_ref3` | 45,222,608 | 4.895015 | 20.792486 | 39 | 815.801985 | 0.815802 | 0.997392 |
| `full_conv_stage_res_cnn_cross_x_ref3` | 46,523,984 | 7.028343 | 20.792486 | 39 | 817.935312 | 0.817935 | 1.000000 |
| `lite_dwres_all_cnn_cross_x_ref3` | 45,158,480 | 3.840147 | 20.792486 | 39 | 814.747117 | 0.814747 | 0.996103 |

## Training Memory

Training memory is read from `train_step_metrics.jsonl`. `Peak allocated` is `cuda_max_mem_allocated_gb`; `Peak reserved` is the maximum logged `cuda_mem_reserved_gb`. All rows use batch size `128`; `cnn_cross_x` uses dynamic `1-3` style references during training, and the other rows use fixed `3` style references. `Normalized memory` is relative to `cnn_cross_x`.

| Configuration | Batch | Train refs | Peak allocated GiB | Peak reserved GiB | Normalized memory |
|---|---:|---|---:|---:|---:|
| `cnn_tokenwise_cross` | 128 | `3` | 22.572310 | 22.777344 | 0.999727 |
| `cnn_global_mean` | 128 | `3` | 22.127525 | 22.287109 | 0.980031 |
| `cnn_global_cls` | 128 | `3` | 22.663972 | 22.921875 | 1.003840 |
| `vit_tokenwise_cross` | 128 | `3` | 22.520816 | 22.695312 | 0.997445 |
| `vit_global_mean` | 128 | `3` | 22.076031 | 22.296875 | 0.977750 |
| `vit_global_cls` | 128 | `3` | 23.977640 | 24.308594 | 1.087196 |
| `cnn_cross_mlp_residual` | 128 | `3` | 22.832003 | 22.982422 | 1.011226 |
| `cnn_cross_mlp_x` | 128 | `3` | 22.832003 | 23.001953 | 1.011226 |
| `cnn_cross_noise` | 128 | `3` | 22.578440 | 22.763672 | 1.000000 |
| `cnn_cross_velocity` | 128 | `3` | 22.578439 | 22.724609 | 1.000000 |
| `cnn_cross_x` | 128 | `1-3` | 22.578439 | 22.800781 | 1.000000 |
| `cnn_cross_x_perc0002` | 128 | `3` | 24.796988 | 24.990234 | 1.098260 |
| `lite_dwres_cnn_cross_x_ref3` | 128 | `3` | 22.097603 | 22.289062 | 0.978702 |
| `full_conv_stage_res_cnn_cross_x_ref3` | 128 | `3` | 22.572310 | 22.740234 | 0.999727 |
| `lite_dwres_all_cnn_cross_x_ref3` | 128 | `3` | 23.346656 | 23.576172 | 1.034024 |

## Metrics Files

| Configuration | Metrics file |
|---|---|
| `cnn_tokenwise_cross` | `checkpoints/xpred_20260601_204907_cnn_tokenwise_cross/eval_full_val_latest_ref3_first3/metrics.json` |
| `cnn_global_mean` | `checkpoints/xpred_20260601_204931_cnn_global_mean/eval_full_val_latest_ref3_first3/metrics.json` |
| `cnn_global_cls` | `checkpoints/xpred_20260603_134809_cnn_global_cls/eval_full_val_latest_ref3_first3/metrics.json` |
| `vit_tokenwise_cross` | `checkpoints/xpred_20260602_153201_vit_tokenwise_cross/eval_full_val_latest_ref3_first3/metrics.json` |
| `vit_global_mean` | `checkpoints/xpred_20260602_153043_vit_global_mean/eval_full_val_latest_ref3_first3/metrics.json` |
| `vit_global_cls` | `checkpoints/xpred_20260603_134831_vit_global_cls/eval_full_val_latest_ref3_first3/metrics.json` |
| `cnn_cross_mlp_residual` | `checkpoints/xpred_20260604_135831_cnn + tokenwise_cross + cross_mlp_residual/eval_full_val_latest_ref3_first3/metrics.json` |
| `cnn_cross_mlp_x` | `checkpoints/xpred_20260605_083238_cnn_tokenwise_cross_cross_mlp_x/eval_full_val_latest_ref3_first3/metrics.json` |
| `cnn_cross_noise` | `checkpoints/xpred_20260605_084110_cnn_tokenwise_cross_cross_noise/eval_full_val_latest_ref3_first3/metrics.json` |
| `cnn_cross_velocity` | `checkpoints/xpred_20260606_055508_cnn_tokenwise_cross_cross_velocity/eval_full_val_latest_ref3_first3/metrics.json` |
| `cnn_cross_x` | `checkpoints/xpred_20260606_055630_cnn_tokenwise_cross_cross_x/eval_full_val_latest_ref3_first3/metrics.json` |
| `cnn_cross_x_perc0002` | `checkpoints/xpred_20260607_191626_cnn_tokenwise_cross_cross_x_perc0002/eval_full_val_latest_ref3_first3/metrics.json` |
| `lite_dwres_cnn_cross_x_ref3` | `checkpoints/xpred_20260609_100404_lite_dwres_cnn_tokenwise_cross_cross_x_ref3/eval_full_val_latest_ref3_first3/metrics.json` |
| `full_conv_stage_res_cnn_cross_x_ref3` | `checkpoints/xpred_20260610_230306_full_conv_stage_res_cnn_tokenwise_cross_cross_x_ref3/eval_full_val_latest_ref3_first3/metrics.json` |
| `lite_dwres_all_cnn_cross_x_ref3` | `checkpoints/xpred_20260614_001143_lite_dwres_all_cnn_tokenwise_cross_cross_x_ref3/eval_full_val_latest_ref3_first3/metrics.json` |
