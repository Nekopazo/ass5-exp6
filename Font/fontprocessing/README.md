# fontprocessing

当前主流程是用 GB2312 汉字全集重建 DiffuFont 的 RGB glyph 数据：

1. 从 Unicode CP936 映射表下载 GB2312 区位汉字。
2. 提取 6763 个 GB2312 汉字候选。
3. 对 `ContentFontList.json` 和 `FontList.json` 中所有字体做 cmap 覆盖检查。
4. 取所有字体共同支持的最终字符集，写入 `CharacterData/CharList.json`。
5. 删除旧的 `DataPreparation/Generated/{ContentFont,TrainFonts}` 和 `DataPreparation/LMDB/{ContentFont.lmdb,TrainFont.lmdb}`。
6. 以 RGB PNG 直接渲染 128x128 glyph 图。
7. 重建 `ContentFont.lmdb` 和 `TrainFont.lmdb`。

运行：

```bash
cd /scratch/yangximing/code/ass5-exp6/DiffuFont
conda run -n sg3 python fontprocessing/rebuild_gb2312_dataset.py \
  --project-root /scratch/yangximing/code/ass5-exp6/DiffuFont \
  --num-workers 48
```

主要输出：

- `fontprocessing/charsets/GB2312.TXT`
- `fontprocessing/charsets/gb2312_hanzi.txt`
- `fontprocessing/charsets/gb2312_common_all_fonts.txt`
- `fontprocessing/charsets/gb2312_common_report.json`
- `CharacterData/CharList.json`
- `DataPreparation/Generated/ContentFont`
- `DataPreparation/Generated/TrainFonts`
- `DataPreparation/LMDB/ContentFont.lmdb`
- `DataPreparation/LMDB/TrainFont.lmdb`
