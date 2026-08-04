# Panorama Depth 实验过程记录

说明：本记录按实验推进顺序整理，不只保留最终好结果，也记录中间失败结果、现象观察和原因分析。评价指标为 AbsRel、RMSE、delta1，其中 AbsRel/RMSE 越低越好，delta1 越高越好。

## 1. 实验目标确定

| 项目 | 内容 |
|---|---|
| 实验目的 | 验证“将全景图切成 perspective views，分别估计单目深度，再融合回全景深度”的方法是否优于直接对整张全景图估计深度。 |
| Baseline | DA3 direct panorama depth，即直接把全景图输入 Depth Anything 3 得到全景深度。 |
| 我们的方法 | Perspective depth fusion / DreamScene360 PanoGeo，即从全景图采样多个 perspective views，估计局部深度，再投回全景坐标融合。 |
| 评价指标 | AbsRel、RMSE、delta1，参考 DAP 论文 Table 3 的 panoramic metric depth 评价方式。 |
| 数据集 | Matterport3D、Stanford2D3D、Deep360。 |

## 2. 第一轮：建立 direct baseline

| 项目 | 内容 |
|---|---|
| 实验设置 | 对 Matterport3D、Stanford2D3D、Deep360 的全景 RGB 直接运行 DA3，得到 direct panorama depth。 |
| 现象观察 | Matterport3D 和 Stanford2D3D 可以作为稳定 baseline；Deep360 的 raw metric 结果非常差，delta1 接近 0。 |
| 初步原因 | DA3 输出虽然看起来像 metric depth，但在不同数据集上的尺度并不稳定。Deep360 的 GT 深度范围较大，预测深度尺度明显偏小。 |
| 后续处理 | 保留 direct 作为 baseline，同时加入 scale diagnostics，检查预测深度和 GT 的尺度差异。 |

## 3. 第二轮：直接做 perspective fusion / PanoGeo

| 项目 | 内容 |
|---|---|
| 实验设置 | 将全景图采样成多个 perspective views，对每个 view 运行 DA3 单目深度，再通过 PanoGeo / fusion 投回全景坐标。早期测试包括 P24-I100，即 24 个 perspective views、100 次 PanoGeo 优化。 |
| 现象观察 | Matterport3D 上有改善趋势；Stanford2D3D 上结果不稳定，AbsRel 有时下降，但 RMSE 或 delta1 可能变差；Deep360 raw metric 下基本失败。 |
| 失败原因 | 单目深度的尺度不可靠，每个 perspective view 的尺度可能不同。融合过程可以补充多视角几何信息，但如果尺度没有对齐，投回全景后会产生错误。 |
| 后续处理 | 增加尺度诊断和对齐方式测试，包括 median-align、fixed-scale calibration、overlap alignment。 |

## 4. 第三轮：检查 Stanford2D3D 数据是否选错

| 项目 | 内容 |
|---|---|
| 实验设置 | 检查 manifest 中 Stanford2D3D 的 RGB/depth 尺寸和路径。 |
| 现象观察 | 早期 manifest 中部分 Stanford2D3D 图像是 1080x1080。 |
| 失败原因 | 1080x1080 是局部 perspective image，不是 2:1 equirectangular panorama，和 DAP Table 3 的 panoramic depth benchmark 口径不一致。 |
| 修正方式 | 改为选择 Stanford2D3D 中 `pano/rgb/*equirectangular*` 对应的 2048x4096 全景 RGB 和匹配的全景 depth。 |
| 修正后结果 | 数据口径变正确，后续 Stanford2D3D 结果才可以用于 panoramic depth 评价。 |

## 5. 第四轮：分析 Deep360 raw metric 失败

| 项目 | 内容 |
|---|---|
| 实验设置 | 对 Deep360 跑 direct DA3 和 PanoGeo/fusion，并同时保存预测结果，检查预测深度中位数与 GT 深度中位数。 |
| 现象观察 | raw metric 下 direct 和 fusion 的 delta1 都接近 0。PanoGeo raw 结果为 AbsRel 0.9299、RMSE 23.7697、delta1 0.0000。 |
| 失败原因 | 主要是尺度问题。Deep360 的 GT 深度尺度与 DA3/PanoGeo 输出尺度差异很大，导致虽然结构可能有部分正确，但 metric 指标直接崩掉。 |
| 修正方式 | 使用 calibration samples 估计固定尺度系数，即 fixed-scale calibration。 |
| 修正后结果 | Deep360 fixed-scale 后，PanoGeo 相比 direct 明显提升：AbsRel 2.2448 -> 0.6195，RMSE 18.5921 -> 12.2246，delta1 0.1528 -> 0.3002。 |

## 6. 第五轮：纯 PanoGeo 对 Stanford2D3D 仍不稳定

| 项目 | 内容 |
|---|---|
| 实验设置 | 在修正 Stanford2D3D 数据口径后，继续测试 P24-I100 和 P60-I300 的 PanoGeo/fusion。 |
| 现象观察 | PanoGeo 提供的多视角信息能改善 AbsRel，但 Stanford2D3D 上 RMSE 和 delta1 仍可能不稳定。 |
| 失败原因 | Stanford2D3D 室内场景结构复杂，局部 perspective 深度投回全景时会引入局部误差；纯 fusion 容易破坏 direct depth 中较稳定的全局结构。 |
| 后续处理 | 不再只使用纯 PanoGeo，而是引入 direct-guided blend：保留 direct panorama depth 的全局稳定性，同时融合 PanoGeo 的多视角几何修正。 |

## 7. 第六轮：加入 direct-guided blend

| 项目 | 内容 |
|---|---|
| 实验设置 | 将 direct panorama depth 与 PanoGeo/fusion depth 做加权融合。Matterport3D 和 Stanford2D3D 使用 P60-I300，即 60 个 perspective views、300 次 PanoGeo 优化，并使用 alpha=0.4 的 direct-guided blend。 |
| 现象观察 | Matterport3D 和 Stanford2D3D 的平均指标都稳定优于 direct baseline。 |
| 原因分析 | Direct depth 提供稳定的全局深度结构，PanoGeo/fusion 提供局部多视角几何补充。两者融合后可以减少纯 fusion 的局部错误，也能修正 direct depth 的部分结构误差。 |
| 最终结果 | Matterport3D 和 Stanford2D3D 三项指标全部提升。 |

## 8. 最终结果表

| Dataset | Baseline Method | Baseline AbsRel | Baseline RMSE | Baseline delta1 | Final Method | Final AbsRel | Final RMSE | Final delta1 | Change |
|---|---|---:|---:|---:|---|---:|---:|---:|---|
| Matterport3D | DA3-Direct-FixedScale-Calib10 | 0.4456 | 0.8291 | 0.4503 | DA3-DirectGuided-PanoGeo-P60-I300-Alpha04 | 0.3680 | 0.7376 | 0.5259 | 三项均提升 |
| Stanford2D3D | DA3-Direct-FixedScale-Calib10 | 0.2537 | 0.7103 | 0.6286 | DA3-DirectGuided-PanoGeo-P60-I300-Alpha04 | 0.2136 | 0.6794 | 0.6397 | 三项均提升 |
| Deep360 | DA3-Direct-Deep360-FixedScale-Calib10 | 2.2448 | 18.5921 | 0.1528 | DreamScene360-PanoGeo-DA3-P24-I100-FixedScale-Calib10 | 0.6195 | 12.2246 | 0.3002 | 三项均提升 |

## 9. 最终提升幅度

| Dataset | AbsRel | RMSE | delta1 | 说明 |
|---|---:|---:|---:|---|
| Matterport3D | 降低约 17.4% | 降低约 11.0% | 提升 7.56 个百分点 | 室内全景上提升明显。 |
| Stanford2D3D | 降低约 15.8% | 降低约 4.4% | 提升 1.11 个百分点 | direct baseline 较强，fusion 带来稳定小幅提升。 |
| Deep360 | 降低约 72.4% | 降低约 34.2% | 提升 14.74 个百分点 | fixed-scale 后提升最明显，说明尺度修正很关键。 |

## 10. 可放入报告的过程性结论

实验过程中最主要的问题不是 perspective fusion 完全无效，而是单目深度的 metric scale 不稳定。早期 raw metric 结果，尤其 Deep360，显示 delta1 接近 0，说明预测深度与 GT 的尺度严重不匹配。经过数据检查和尺度校准后，Deep360 的结果明显改善。

Stanford2D3D 的问题主要是纯 PanoGeo/fusion 容易引入局部误差。因此后续加入 direct-guided blend，用 direct panorama depth 保持全局结构稳定，用 PanoGeo 提供局部几何修正。最终 Matterport3D、Stanford2D3D 和 Deep360 三个数据集上都得到了优于 direct baseline 的结果。
