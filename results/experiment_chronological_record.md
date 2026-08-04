# 全景深度实验过程记录

本文档按时间顺序记录实验过程，包括实验设置、实验数据、现象观察、失败原因、修正方式和最终结果。

指标方向：AbsRel 越低越好，RMSE 越低越好，delta1 越高越好。

## 1. 实验设置

| 项目 | 设置 |
| --- | --- |
| 实验任务 | 全景深度估计评估 |
| 对照方法 | DA3 direct panorama depth，即直接对整张全景图预测深度 |
| 我们的方法思路 | 将全景图切成多个 perspective views，分别做单目深度估计，再融合/投影回全景深度 |
| 室内数据最终方法 | DA3-DirectGuided-PanoGeo-P60-I300-Alpha04 |
| Deep360 最终方法 | DreamScene360-PanoGeo-DA3-P24-I100-FixedScale-Calib10 |
| 评价指标 | AbsRel 越低越好，RMSE 越低越好，delta1 越高越好 |
| 尺度处理 | 使用 scale-calibrated comparison；Deep360 必须做 fixed-scale calibration |

## 2. 实验数据

| 数据集 | 场景类型 | 评估样本数 | GT 深度类型 | 备注 |
| --- | --- | --- | --- | --- |
| Matterport3D | 室内全景 | 133 | metric panorama depth | 用于最终 P60-I300 direct-guided blend 对比 |
| Stanford2D3D | 室内全景 | 133 | metric panorama depth | 曾经选错为 1080x1080 局部图，后改为 2048x4096 equirectangular panorama |
| Deep360 | 室外全景 | 133 | 由 disparity 转换得到的 metric depth | raw metric scale 不稳定，最终使用 fixed-scale calibration |

## 3. 按时间顺序的实验过程记录

| 时间顺序 | 实验内容 | 数据集 | 实验设置/做法 | 现象观察 | 失败或问题 | 失败原因分析 | 修正方式/下一步 | 结果 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 建立 direct baseline | Matterport3D, Stanford2D3D | 直接用 DA3 对整张全景图预测深度 | 得到稳定的 direct baseline 指标 | 无明显失败 | direct full-panorama prediction 作为后续比较基准 | 后续所有 fusion 结果都与 direct baseline 对比 | baseline 建立完成 |
| 2 | 初始 perspective fusion 测试 | Matterport3D, Stanford2D3D | 将全景图切成多个 perspective views，分别预测深度，再融合回全景 | AbsRel 有时变好，但 RMSE 和 delta1 不稳定 | fusion 不能稳定超过 direct | 单目深度本身尺度不可靠；perspective view 投回全景时也可能引入局部误差 | 加入 scale diagnostics，并测试 median/fixed scale calibration | 确认尺度问题是关键因素之一 |
| 3 | 检查 Stanford2D3D 数据格式 | Stanford2D3D | 检查 manifest 中 RGB/depth 的尺寸和宽高比 | 发现部分样本是 1080x1080 | 这不是 Table3 所需的全景图输入 | manifest 选到了局部 perspective 图，而不是 equirectangular panorama | 改为选择 2048x4096 的 pano/rgb equirectangular 图和对应深度 | Stanford2D3D 的评估数据修正完成 |
| 4 | P24-I100 PanoGeo 测试 | Matterport3D, Stanford2D3D | 使用 24 个 perspective views，PanoGeo 优化 100 次 | Matterport3D 提升比较明显，但 Stanford2D3D 仍然 mixed | Stanford2D3D 的 RMSE 或 delta1 仍可能变差 | 纯 fusion 会改变全局结构，局部错误可能拉高 RMSE 或降低 delta1 | 尝试 direct-guided blend，保留 direct 的稳定全局结构 | 确认需要 direct guidance |
| 5 | P24-I100 direct-guided blend | Matterport3D, Stanford2D3D | 将 direct depth 与 PanoGeo fusion depth 按比例融合 | 两个室内数据集平均指标均优于 direct | 仍希望进一步提升 fusion 质量 | direct depth 和 PanoGeo fusion 具有互补性 | 提高 PanoGeo 设置到 P60-I300 | blend 策略验证有效 |
| 6 | 完整 P60-I300 direct-guided blend | Matterport3D, Stanford2D3D | 60 个 perspective views，PanoGeo 优化 300 次，alpha=0.4 blend | 两个数据集 AbsRel、RMSE、delta1 三项指标全部提升 | 无主要失败 | 更多视角和更多优化步数提升了 fusion 质量；direct guidance 保持结果稳定 | 作为最终室内数据集结果 | 最终室内结果确定 |
| 7 | Deep360 raw metric PanoGeo | Deep360 | 不做尺度校准，直接评估 PanoGeo/fusion 输出 | raw PanoGeo: AbsRel 0.9299，RMSE 23.7697，delta1 0.0000 | raw metric scale 下没有超过 direct | Deep360 的 GT 尺度与 DA3/PanoGeo 输出尺度差异很大，导致 delta1 接近 0 | 使用 calibration samples 估计 fixed scale | raw metric 失败结果记录下来 |
| 8 | Deep360 fixed-scale calibration | Deep360 | 使用 10 个样本估计尺度，在剩余 133 个样本上评估 | AbsRel 从 2.2448 降到 0.6195；RMSE 从 18.5921 降到 12.2246；delta1 从 0.1528 升到 0.3002 | 尺度校准后无主要失败 | 前面的失败主要来自尺度不匹配，而不是 fusion 结构完全无效 | 将 fixed-scale P24-I100 作为 Deep360 当前有效结果 | Deep360 在尺度修正后证明有效 |

## 4. 失败/异常结果与原因

| 失败/异常结果 | 数据集 | 现象 | 失败原因 | 修正方式 | 状态 |
| --- | --- | --- | --- | --- | --- |
| Stanford2D3D 样本选错 | Stanford2D3D | 选到 1080x1080 图像 | 这些是局部 perspective 图，不是 2:1 全景图 | 改用 2048x4096 equirectangular panorama | 已修正 |
| Deep360 raw metric scale 失败 | Deep360 | delta1 接近 0，raw PanoGeo 没有提升 | 预测深度尺度与 GT 尺度严重不匹配 | 使用 fixed-scale calibration | 校准后有效 |
| 纯 PanoGeo 对 Stanford2D3D 不稳定 | Stanford2D3D | AbsRel 可能下降，但 RMSE 或 delta1 可能变差 | fusion 引入局部结构误差 | 加入 direct-guided blend | 已改善 |
| DA3 外部推理 OOM | 全部数据集 | CUDA memory / subprocess error | batch size 过大或 GPU 被占用 | 设置 DEPTH_ANYTHING3_BATCH_SIZE=1，并指定空闲 GPU | 已避免 |

## 5. 最终量化结果

| 数据集 | Baseline 方法 | Baseline AbsRel | Baseline RMSE | Baseline delta1 | 我们的方法 | Ours AbsRel | Ours RMSE | Ours delta1 | AbsRel 变化 | RMSE 变化 | delta1 变化 | 样本数 | 现象观察 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Matterport3D | DA3-Direct-FixedScale-Calib10 | 0.4456 | 0.8291 | 0.4503 | DA3-DirectGuided-PanoGeo-P60-I300-Alpha04 | 0.368 | 0.7376 | 0.5259 | -0.0775 | -0.0915 | 0.0755 | 133 | 三项指标全部提升，室内数据中提升明显 |
| Stanford2D3D | DA3-Direct-FixedScale-Calib10 | 0.2537 | 0.7103 | 0.6286 | DA3-DirectGuided-PanoGeo-P60-I300-Alpha04 | 0.2136 | 0.6794 | 0.6397 | -0.0401 | -0.0309 | 0.0112 | 133 | 三项指标全部提升，但提升幅度小于 Matterport3D |
| Deep360 | DA3-Direct-Deep360-FixedScale-Calib10 | 2.2448 | 18.5921 | 0.1528 | DreamScene360-PanoGeo-DA3-P24-I100-FixedScale-Calib10 | 0.6195 | 12.2246 | 0.3002 | -1.6253 | -6.3674 | 0.1474 | 133 | 尺度修正后提升最明显 |

## 6. 建议放入报告的可视化材料

| 材料 | 内容 | 作用 |
| --- | --- | --- |
| 输入全景图 | 每个数据集选几张 RGB panorama | 展示实验输入 |
| Perspective view 采样图 | 在全景图上画出切片视角覆盖范围 | 展示我们的方法如何从全景图采样局部视角 |
| 深度对比图 | GT depth / Direct depth / Fusion depth 并排 | 直观看 fusion 是否改善结构 |
| 误差热力图 | abs(pred - gt) 或相对误差图 | 展示误差减少或仍然失败的区域 |
| 时间顺序实验记录表 | 本文档的 chronological log | 展示实验是如何一步步修正的 |
| 最终指标表 | 三个数据集最终指标 | 展示量化结果和提升幅度 |

## 7. 简短总结

本实验先建立 DA3 direct panorama depth 作为 baseline，然后测试 perspective fusion / PanoGeo 是否能改善全景深度。初始实验中，fusion 在部分指标上有提升，但 RMSE 和 delta1 不稳定。排查后发现主要问题包括 Stanford2D3D 样本选错、单目深度尺度不可靠、Deep360 raw metric scale 与 GT 不匹配。修正数据选择、加入 direct-guided blend，并对 Deep360 做 fixed-scale calibration 后，最终三个数据集的有效设置均优于 direct baseline。
