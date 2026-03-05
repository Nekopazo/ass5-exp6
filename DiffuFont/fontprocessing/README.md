# fontprocessing

基于“字表（默认可传入 3500 常用简体字）” + `CLIP ViT-B/32` 图像特征 + `KMeans` 聚类，自动生成 `400` 个参考字及其 cluster 桶映射。

## 目标流程

1. 读取指定字表（或默认生成 `GB2312` 汉字集合）
2. 用 base font (`.ttf`) 渲染 `224x224` 字图
3. 用 CLIP image encoder 提取 embedding（`openai/clip-vit-base-patch32`）
4. 对 embedding 做 KMeans 聚类，默认扫描 `K = 12/16/20/24/28`
5. 每个 cluster 选代表字
- 先选离 centroid 最近的 1 个中心字
- 再用 farthest-point sampling 选其余字，覆盖 cluster 内结构变化
6. 合并得到 `400` 字 reference
7. 保存 cluster id 作为桶

输出核心文件：
- `reference_400.txt`
- `reference_cluster.json`

## 安装

```bash
cd /scratch/yangximing/fontprocessing
pip install -r requirements.txt
```

## 运行

```bash
cd /scratch/yangximing/fontprocessing
python run_pipeline.py \
  --font /path/to/base_font.ttf \
  --charset-file /scratch/yangximing/fontprocessing/charsets/common_simplified_3500.txt \
  --out-dir outputs \
  --k-list 12 16 20 24 28 \
  --target-total 400
```

常用参数：
- `--clip-model` 默认 `openai/clip-vit-base-patch32`
- `--charset-file` 可传入外部字表（支持每行1字或整行连续字）
- `--font-size` 默认 `180`
- `--batch-size` 默认 `128`
- `--device` 默认自动选择 `cuda`/`cpu`

## 输出目录结构

```text
outputs/
  gb2312_hanzi.txt
  gb2312_clip_embeds.npy
  k_search_metrics.json
  best_k.json
  reference_400.txt
  reference_cluster.json
  k_24/
    reference_400.txt
    reference_cluster.json
    metrics.json
  k_32/
  k_40/
  k_50/
  k_64/
```

## K 选择逻辑

每个 K 计算：
- `p95_d`
- `min_cluster_size`
- `max_min_ratio`

默认自动选择规则：
1. `min_cluster_size` 更大优先
2. `p95_d` 更小优先
3. `max_min_ratio` 更小优先

可通过 `outputs/k_search_metrics.json` 手动判断，通常会落在 `K=40` 或 `K=50`。
