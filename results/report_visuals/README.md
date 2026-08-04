# Report Visuals

已经本地生成：

| 文件 | 用途 |
| --- | --- |
| `metric_bars_three_datasets.png` | 三个数据集的 AbsRel、RMSE、delta1 指标柱状图 |

服务器恢复后生成：

| 文件 | 用途 |
| --- | --- |
| `matterport3d_depth_comparison.png` | Matterport3D 的 RGB / GT / Direct / Ours / Error 对比图 |
| `stanford2d3d_depth_comparison.png` | Stanford2D3D 的 RGB / GT / Direct / Ours / Error 对比图 |
| `deep360_depth_comparison.png` | Deep360 的 RGB / GT / Direct / Ours / Error 对比图 |

PowerShell 上传脚本和运行命令：

```powershell
scp `
  "C:\Users\克斯维尔\Desktop\projects\DreamScene360\scripts\visualize_report_depth_results.py" `
  "C:\Users\克斯维尔\Desktop\projects\DreamScene360\results\report_visuals\run_server_depth_visuals.sh" `
  wangqq:/mnt/data/wangqq/DreamScene360/scripts/
```

服务器里运行：

```bash
cd /mnt/data/wangqq/DreamScene360

mv scripts/run_server_depth_visuals.sh results/report_visuals/run_server_depth_visuals.sh
bash results/report_visuals/run_server_depth_visuals.sh
```

生成完后从服务器下载：

```powershell
scp wangqq:/mnt/data/wangqq/DreamScene360/results/report_visuals/*.png `
  "C:\Users\克斯维尔\Desktop\projects\DreamScene360\results\report_visuals\"
```
