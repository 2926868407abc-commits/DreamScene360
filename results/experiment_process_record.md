# DreamScene360 全景深度实验过程记录

本文档按实验推进顺序记录全景深度实验。重点记录每一步做了什么、看到什么现象、哪里失败、为什么失败，以及后续如何修正。

指标方向如下：

| 指标 | 含义 | 判断方式 |
| --- | --- | --- |
| AbsRel | 相对深度误差 | 越低越好 |
| RMSE | 均方根深度误差 | 越低越好 |
| delta1 | 深度预测在 1.25 阈值内的准确率 | 越高越好 |

## 1. 实验设置

### 1.1 总体目标

| 项目 | 内容 |
| --- | --- |
| 实验目标 | 验证 perspective-based PanoGeo / fusion 是否能改善全景深度估计 |
| Baseline | DA3 direct panorama depth，即直接对整张全景图预测深度 |
| 我们的方法 | 将全景图切成 perspective views，分别估计单目深度，再融合/投影回全景深度 |
| 核心对比 | direct panorama depth vs perspective-based fusion depth |

### 1.2 最终采用的有效设置

| 数据集类型 | 最终方法 | 说明 |
| --- | --- | --- |
| 室内数据集 | DA3-DirectGuided-PanoGeo-P60-I300-Alpha04 | 用于 Matterport3D 和 Stanford2D3D |
| Deep360 | DreamScene360-PanoGeo-DA3-P24-I100-FixedScale-Calib10 | Deep360 raw metric scale 不稳定，因此使用 fixed-scale calibration 后的结果 |

## 2. 实验数据

### 2.1 数据集信息

| 数据集 | 类型 | 评估样本数 | GT 深度 |
| --- | --- | ---: | --- |
| Matterport3D | 室内全景 | 133 | metric panorama depth |
| Stanford2D3D | 室内全景 | 133 | metric panorama depth |
| Deep360 | 室外全景 | 133 | disparity 转换得到的 metric depth |

### 2.2 数据检查备注

| 数据集 | 检查结果 | 备注 |
| --- | --- | --- |
| Matterport3D | 1024x2048，宽高比 2:1 | 本身就是 panorama，可直接使用 |
| Stanford2D3D | 修正后使用 2048x4096，宽高比 2:1 | 早期曾选错为 1080x1080 局部图 |
| Deep360 | 使用解压后的全景 RGB 和 depth/disparity 文件 | raw metric scale 不稳定，需要单独做尺度校准 |

## 3. 实验时间线总览

| 顺序 | 阶段 | 主要目的 |
| ---: | --- | --- |
| 1 | 小样本 direct / fusion 测试 | 确认流程能跑通，并观察 fusion 是否有潜力 |
| 2 | 数据集检查 | 确认输入是否是真正 panorama |
| 3 | P24-I100 PanoGeo 排查 | 测试纯 fusion 是否稳定优于 direct |
| 4 | Direct-guided blend | 结合 direct 的稳定性和 fusion 的局部几何 |
| 5 | P60-I300 主线 | 提高 views 数量和优化步数，得到室内最终结果 |
| 6 | Deep360 raw metric 排查 | 检查室外数据集失败原因 |
| 7 | Deep360 fixed-scale 修正 | 通过尺度校准得到 Deep360 有效结果 |

## 4. 阶段一：小样本初始评测

这一阶段只用于确认代码流程和评估链路是否可用。样本数只有 5，因此不作为最终结论。

### 4.1 Matterport3D 小样本结果

| 方法 | 样本数 | AbsRel | RMSE | delta1 |
| --- | ---: | ---: | ---: | ---: |
| DA3 direct local samples | 5 | 0.4620 | 1.0850 | 0.1164 |
| DA3 perspective fusion, metric | 5 | 0.3368 | 0.9526 | 0.2702 |
| DA3 fusion + median align | 5 | 0.2023 | 0.6042 | 0.6368 |

### 4.2 Matterport3D 小样本观察

| 现象 | 判断 |
| --- | --- |
| fusion 的三项指标都好于 direct | perspective fusion 对 Matterport3D 有潜力 |
| median align 后进一步变好 | 尺度对齐会明显影响指标 |

### 4.3 Stanford2D3D 小样本结果

| 方法 | 样本数 | AbsRel | RMSE | delta1 |
| --- | ---: | ---: | ---: | ---: |
| DA3 direct local samples | 5 | 0.6013 | 1.6680 | 0.0093 |
| DA3 perspective fusion, metric | 5 | 0.5048 | 2.3993 | 0.0739 |
| DA3 fusion + median align | 5 | 0.3005 | 1.6099 | 0.4596 |

### 4.4 Stanford2D3D 小样本观察

| 现象 | 判断 |
| --- | --- |
| raw metric fusion 的 AbsRel 和 delta1 变好，但 RMSE 变差 | fusion 有效果，但尺度或局部结构不稳定 |
| median align 后三项指标明显改善 | Stanford2D3D 也受尺度影响很大 |

### 4.5 阶段一结论

| 结论 | 后续动作 |
| --- | --- |
| perspective fusion 不是无效的 | 继续做 PanoGeo / fusion 排查 |
| raw metric scale 下结果不稳定 | 后续加入尺度诊断和尺度校准 |
| Stanford2D3D 结果异常较多 | 单独检查 Stanford2D3D 数据是否选对 |

## 5. 阶段二：数据集检查与 Stanford2D3D 修正

这一阶段排查评估数据是否符合全景深度 benchmark。主要问题出现在 Stanford2D3D。

### 5.1 Stanford2D3D 发现的问题

| 检查项 | 修正前现象 | 问题 |
| --- | --- | --- |
| 图像尺寸 | 1080x1080 | 不是 2:1 全景图 |
| 图像类型 | 局部 perspective 图 | 不符合 panorama benchmark 输入 |
| 文件路径 | 局部 `data/rgb` 图像 | 没有使用 equirectangular panorama |

### 5.2 Stanford2D3D 修正方式

| 修正项 | 修正后结果 |
| --- | --- |
| 图像尺寸 | 改为 2048x4096 |
| 宽高比 | 2:1 |
| 文件路径 | 使用 `pano/rgb/*equirectangular*` 和对应 panorama depth |

### 5.3 数据检查结论

| 数据集 | 结论 |
| --- | --- |
| Stanford2D3D | 早期异常结果不能完全归因于方法，数据选择也有问题；修正后统一使用 panorama 图 |
| Matterport3D | 本身是 1024x2048 的 2:1 panorama，不需要修正 |

## 6. 阶段三：P24-I100 PanoGeo 排查

修正数据后，先使用较快的 P24-I100 设置测试 PanoGeo。这个阶段的重点是判断纯 fusion 的问题在哪里。

### 6.1 P24-I100 设置

| 参数 | 设置 |
| --- | --- |
| perspective views 数量 | 24 |
| PanoGeo 优化步数 | 100 |
| 评估方式 | per-image median 排查 |
| 目的 | 判断纯 fusion 是否稳定优于 direct |

### 6.2 Matterport3D P24-I100 结果

| 方法 | 样本数 | AbsRel | RMSE | delta1 |
| --- | ---: | ---: | ---: | ---: |
| DA3 direct, per-image median | 143 | 0.3347 | 0.7263 | 0.5137 |
| PanoGeo P24-I100, per-image median | 143 | 0.2105 | 0.7026 | 0.6933 |

### 6.3 Matterport3D P24-I100 观察

| 指标变化 | 说明 |
| --- | --- |
| AbsRel 下降 | 相对深度误差变小 |
| RMSE 下降 | 整体深度误差变小 |
| delta1 上升 | 准确率提高 |

### 6.4 Stanford2D3D P24-I100 结果

| 方法 | 样本数 | AbsRel | RMSE | delta1 |
| --- | ---: | ---: | ---: | ---: |
| DA3 direct, per-image median | 143 | 0.2179 | 0.6798 | 0.6878 |
| PanoGeo P24-I100, per-image median | 143 | 0.2068 | 0.7210 | 0.6341 |

### 6.5 Stanford2D3D P24-I100 观察

| 指标变化 | 说明 |
| --- | --- |
| AbsRel 下降 | 有一部分相对误差被改善 |
| RMSE 上升 | 存在局部大误差，拉高了 RMSE |
| delta1 下降 | 满足 1.25 阈值的像素比例减少 |

### 6.6 阶段三失败原因

| 现象 | 原因分析 |
| --- | --- |
| Matterport3D 上纯 PanoGeo 效果较好 | 室内结构较适合多视角几何融合 |
| Stanford2D3D 上 RMSE/delta1 变差 | 纯 fusion 改善了一部分相对误差，但可能引入局部结构误差 |
| 纯 fusion 不够稳定 | direct depth 的全局结构更稳定，fusion 的局部几何更强 |

### 6.7 阶段三修正方向

| 修正方向 | 目的 |
| --- | --- |
| 加入 direct-guided blend | 保留 direct 全局稳定性 |
| 继续使用 PanoGeo | 利用 perspective fusion 的局部几何信息 |

## 7. 阶段四：Direct-guided Blend

Direct-guided blend 的目的，是保留 direct panorama depth 的全局稳定性，同时利用 PanoGeo / perspective fusion 的局部几何补充。

### 7.1 Blend 思路

| 组成部分 | 作用 |
| --- | --- |
| direct depth | 提供稳定的全局结构 |
| PanoGeo fusion depth | 提供多视角局部几何补充 |
| blend | 将二者结合，减少纯 fusion 的局部错误 |

### 7.2 Matterport3D P24-I100 blend 结果

| 方法 | 样本数 | AbsRel | RMSE | delta1 |
| --- | ---: | ---: | ---: | ---: |
| DA3 direct fixed-scale | 133 | 0.4456 | 0.8291 | 0.4503 |
| P24-I100 direct-guided blend | 133 | 0.3674 | 0.7483 | 0.5153 |

### 7.3 Stanford2D3D P24-I100 blend 结果

| 方法 | 样本数 | AbsRel | RMSE | delta1 |
| --- | ---: | ---: | ---: | ---: |
| DA3 direct fixed-scale | 133 | 0.2537 | 0.7103 | 0.6286 |
| P24-I100 direct-guided blend | 133 | 0.2169 | 0.6909 | 0.6402 |

### 7.4 阶段四结论

| 结论 | 说明 |
| --- | --- |
| 两个室内数据集三项指标都提升 | blend 比纯 PanoGeo 更稳定 |
| Stanford2D3D 的 RMSE/delta1 问题得到缓解 | direct guidance 对稳定结构有帮助 |
| 可以继续提高 PanoGeo 设置 | 后续跑 P60-I300 完整设置 |

## 8. 阶段五：最终室内主线 P60-I300 + Direct-guided Blend

P60-I300 是当前室内数据集最终主线：60 个 perspective views，PanoGeo 优化 300 次，并使用 alpha=0.4 的 direct-guided blend。

### 8.1 P60-I300 设置

| 参数 | 设置 |
| --- | --- |
| perspective views 数量 | 60 |
| PanoGeo 优化步数 | 300 |
| blend alpha | 0.4 |
| 适用数据集 | Matterport3D, Stanford2D3D |

### 8.2 Matterport3D 最终室内结果

| 方法 | 样本数 | AbsRel | RMSE | delta1 |
| --- | ---: | ---: | ---: | ---: |
| DA3 direct fixed-scale | 133 | 0.4456 | 0.8291 | 0.4503 |
| P60-I300 direct-guided blend | 133 | 0.3680 | 0.7376 | 0.5259 |

### 8.3 Matterport3D 提升幅度

| 指标 | 变化 | 说明 |
| --- | --- | --- |
| AbsRel | 降低 17.4% | 相对误差明显下降 |
| RMSE | 降低 11.0% | 整体误差下降 |
| delta1 | 提升 7.56 个百分点 | 准确率提高 |

### 8.4 Stanford2D3D 最终室内结果

| 方法 | 样本数 | AbsRel | RMSE | delta1 |
| --- | ---: | ---: | ---: | ---: |
| DA3 direct fixed-scale | 133 | 0.2537 | 0.7103 | 0.6286 |
| P60-I300 direct-guided blend | 133 | 0.2136 | 0.6794 | 0.6397 |

### 8.5 Stanford2D3D 提升幅度

| 指标 | 变化 | 说明 |
| --- | --- | --- |
| AbsRel | 降低 15.8% | 相对误差下降 |
| RMSE | 降低 4.4% | 整体误差小幅下降 |
| delta1 | 提升 1.11 个百分点 | 准确率小幅提高 |

### 8.6 阶段五结论

| 数据集 | 结论 |
| --- | --- |
| Matterport3D | 提升明显，三项指标全部优于 baseline |
| Stanford2D3D | 提升幅度较小，但三项指标方向一致，均优于 baseline |

## 9. 阶段六：Deep360 尺度问题与修正

Deep360 上最开始 raw metric scale 结果失败。这个失败不是简单说明 fusion 无效，而是说明预测深度和 Deep360 GT 深度之间存在严重尺度不匹配。

### 9.1 Deep360 raw metric 结果

| 方法 | 样本数 | AbsRel | RMSE | delta1 |
| --- | ---: | ---: | ---: | ---: |
| DA3 direct raw metric | 143 | 0.8796 | 23.7792 | 0.0006 |
| PanoGeo P60-I300 raw metric | 143 | 0.9299 | 23.7697 | 0.0000 |

### 9.2 Deep360 raw metric 现象

| 现象 | 说明 |
| --- | --- |
| delta1 接近 0 | 几乎没有像素满足 1.25 阈值 |
| RMSE 很大 | Deep360 深度范围大，尺度错误被放大 |
| raw PanoGeo 没有超过 direct | 指标主要被尺度错误主导 |

### 9.3 Deep360 失败原因

| 原因 | 解释 |
| --- | --- |
| 预测深度尺度和 GT 尺度不匹配 | DA3/PanoGeo 输出尺度与 Deep360 GT 深度尺度差异大 |
| raw metric 直接评估不公平 | 没有做尺度校准时，delta1 和 RMSE 会被尺度误差主导 |

### 9.4 Deep360 fixed-scale 修正结果

| 方法 | 样本数 | AbsRel | RMSE | delta1 |
| --- | ---: | ---: | ---: | ---: |
| DA3 direct fixed-scale calib10 | 133 | 2.2448 | 18.5921 | 0.1528 |
| PanoGeo P24-I100 fixed-scale calib10 | 133 | 0.6195 | 12.2246 | 0.3002 |

### 9.5 Deep360 fixed-scale 提升幅度

| 指标 | 变化 | 说明 |
| --- | --- | --- |
| AbsRel | 降低 72.4% | 相对误差大幅下降 |
| RMSE | 降低 34.2% | 整体误差明显下降 |
| delta1 | 提升 14.74 个百分点 | 准确率明显提高 |

### 9.6 阶段六结论

| 结论 | 说明 |
| --- | --- |
| Deep360 raw metric 失败主要来自尺度不匹配 | 不是 PanoGeo 结构完全无效 |
| fixed-scale calibration 后 PanoGeo 明显优于 direct | 三项指标全部提升 |

## 10. 最终结果汇总

### 10.1 最终方法对应关系

| 数据集 | 最终方法 |
| --- | --- |
| Matterport3D | P60-I300 direct-guided blend |
| Stanford2D3D | P60-I300 direct-guided blend |
| Deep360 | P24-I100 fixed-scale PanoGeo |

### 10.2 Matterport3D 最终指标

| 方法 | AbsRel | RMSE | delta1 |
| --- | ---: | ---: | ---: |
| Baseline | 0.4456 | 0.8291 | 0.4503 |
| Ours | 0.3680 | 0.7376 | 0.5259 |

### 10.3 Stanford2D3D 最终指标

| 方法 | AbsRel | RMSE | delta1 |
| --- | ---: | ---: | ---: |
| Baseline | 0.2537 | 0.7103 | 0.6286 |
| Ours | 0.2136 | 0.6794 | 0.6397 |

### 10.4 Deep360 最终指标

| 方法 | AbsRel | RMSE | delta1 |
| --- | ---: | ---: | ---: |
| Baseline | 2.2448 | 18.5921 | 0.1528 |
| Ours | 0.6195 | 12.2246 | 0.3002 |

### 10.5 总体结论

| 数据集 | 是否三项指标都提升 | 备注 |
| --- | --- | --- |
| Matterport3D | 是 | 提升明显 |
| Stanford2D3D | 是 | 提升较小但方向一致 |
| Deep360 | 是 | fixed-scale 后提升最明显 |

## 11. 给老师看的简短结论

前期实验中确实出现过不稳定和错误结果。主要问题有三个：Stanford2D3D 早期选到了 1080x1080 局部 perspective 图，纯 PanoGeo 在 Stanford2D3D 上会出现 AbsRel 变好但 RMSE/delta1 变差，Deep360 在 raw metric scale 下 delta1 接近 0。

后续分别做了修正：Stanford2D3D 改为 2048x4096 equirectangular panorama；室内数据从纯 PanoGeo 改为 direct-guided blend；Deep360 使用 calibration samples 做 fixed-scale comparison。

修正后，Matterport3D、Stanford2D3D 和 Deep360 三个数据集的最终设置都优于 direct baseline。说明 perspective-based PanoGeo fusion 对全景深度估计是有效的，但需要保证输入数据是真正的 panorama，同时要处理好单目深度尺度问题。
