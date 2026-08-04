# 实验部分写法草稿

## 实验设置

本实验的目标是验证基于 perspective views 的全景深度融合方法是否能够提升全景深度估计质量。实验以 DA3 direct panorama depth 作为 baseline，即直接对整张 equirectangular panorama 预测深度；我们的方法则先将全景图采样为多个 perspective views，对每个局部视角分别进行单目深度估计，再通过 PanoGeo / perspective fusion 将局部深度融合回全景深度。

实验在三个全景深度数据集上进行评估：Matterport3D、Stanford2D3D 和 Deep360。其中 Matterport3D 和 Stanford2D3D 为室内全景数据集，Deep360 为室外全景数据集。评价指标采用 AbsRel、RMSE 和 delta1，其中 AbsRel 和 RMSE 越低越好，delta1 越高越好。

| 数据集 | 场景类型 | 评估样本数 | GT 深度类型 |
| --- | --- | ---: | --- |
| Matterport3D | 室内全景 | 133 | metric panorama depth |
| Stanford2D3D | 室内全景 | 133 | metric panorama depth |
| Deep360 | 室外全景 | 133 | disparity 转换得到的 metric depth |

## 实验过程与问题修正

早期实验首先在小样本上测试 direct depth 和 perspective fusion。结果显示，fusion 在 Matterport3D 上可以带来明显改善，但在 Stanford2D3D 和 Deep360 上存在不稳定现象。进一步排查后发现主要有三个问题。

第一，Stanford2D3D 早期样本选择存在错误，部分样本选到了 1080x1080 的局部 perspective 图像，而不是 2:1 的 equirectangular panorama。该问题会导致评估输入与全景深度 benchmark 不一致。后续将 Stanford2D3D 输入修正为 2048x4096 的 panorama 图像。

第二，纯 PanoGeo / perspective fusion 在 Stanford2D3D 上会出现 AbsRel 下降但 RMSE 或 delta1 变差的情况，说明纯 fusion 虽然改善了一部分相对深度误差，但可能引入局部结构错误。为此，后续加入 direct-guided blend，将 direct depth 的全局稳定性和 PanoGeo fusion 的局部几何信息结合起来。

第三，Deep360 在 raw metric scale 下表现异常，delta1 接近 0。该现象说明 DA3/PanoGeo 输出深度与 Deep360 GT 深度之间存在严重尺度不匹配。后续使用 calibration samples 进行 fixed-scale calibration，再在剩余样本上评估。

| 问题 | 数据集 | 现象 | 修正方式 |
| --- | --- | --- | --- |
| 样本选错 | Stanford2D3D | 选到 1080x1080 局部 perspective 图 | 改用 2048x4096 equirectangular panorama |
| 纯 fusion 不稳定 | Stanford2D3D | AbsRel 变好，但 RMSE/delta1 可能变差 | 加入 direct-guided blend |
| raw metric scale 失败 | Deep360 | delta1 接近 0 | 使用 fixed-scale calibration |

## 量化结果

最终结果如下。Matterport3D 和 Stanford2D3D 使用 P60-I300 direct-guided blend，其中 P60 表示采样 60 个 perspective views，I300 表示 PanoGeo 优化 300 次。Deep360 使用 fixed-scale P24-I100 PanoGeo，这是当前验证有效的 Deep360 设置。

| 数据集 | Baseline 方法 | Baseline AbsRel | Baseline RMSE | Baseline delta1 | 我们的方法 | Ours AbsRel | Ours RMSE | Ours delta1 |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| Matterport3D | DA3-Direct-FixedScale-Calib10 | 0.4456 | 0.8291 | 0.4503 | P60-I300 direct-guided blend | 0.3680 | 0.7376 | 0.5259 |
| Stanford2D3D | DA3-Direct-FixedScale-Calib10 | 0.2537 | 0.7103 | 0.6286 | P60-I300 direct-guided blend | 0.2136 | 0.6794 | 0.6397 |
| Deep360 | DA3-Direct-Deep360-FixedScale-Calib10 | 2.2448 | 18.5921 | 0.1528 | P24-I100 fixed-scale PanoGeo | 0.6195 | 12.2246 | 0.3002 |

从结果可以看出，三个数据集上最终方法均优于 direct baseline。Matterport3D 上 AbsRel 从 0.4456 降到 0.3680，RMSE 从 0.8291 降到 0.7376，delta1 从 0.4503 提升到 0.5259。Stanford2D3D 上三项指标也全部提升，但提升幅度小于 Matterport3D。Deep360 在尺度校准后提升最明显，AbsRel 从 2.2448 降到 0.6195，RMSE 从 18.5921 降到 12.2246，delta1 从 0.1528 提升到 0.3002。

| 数据集 | AbsRel 改善 | RMSE 改善 | delta1 改善 | 结论 |
| --- | ---: | ---: | ---: | --- |
| Matterport3D | 降低 17.4% | 降低 11.0% | 提升 7.56 个百分点 | 提升明显 |
| Stanford2D3D | 降低 15.8% | 降低 4.4% | 提升 1.11 个百分点 | 稳定提升 |
| Deep360 | 降低 72.4% | 降低 34.2% | 提升 14.74 个百分点 | 尺度修正后提升最明显 |

## 可视化结果应该怎么放

可视化部分不需要放很多花哨图片，但需要覆盖“输入是什么、方法做了什么、结果哪里变好、失败怎么修正”这四件事。建议至少放以下四组图。

### 图 1：方法流程可视化

建议内容：输入 panorama、perspective view 采样示意、局部 depth prediction、融合后的 panorama depth。

作用：说明方法不是直接对全景图预测，而是先切 perspective views，再融合回 panorama depth。

### 图 2：三数据集最终深度对比

建议每个数据集选 1 个代表样本，排成 3 行，每行包含：

| 列 | 内容 |
| --- | --- |
| 1 | RGB panorama |
| 2 | GT depth |
| 3 | DA3 direct depth |
| 4 | Ours / fusion depth |
| 5 | direct error map |
| 6 | ours error map |

作用：直观看出我们的方法相对于 direct baseline 的误差是否减少。

### 图 3：失败结果与修正对比

建议放 Deep360 的 raw metric 和 fixed-scale 结果：

| 列 | 内容 |
| --- | --- |
| 1 | RGB panorama |
| 2 | raw metric PanoGeo depth |
| 3 | fixed-scale PanoGeo depth |
| 4 | GT depth |
| 5 | error map |

作用：说明 Deep360 早期失败不是方法完全无效，而是尺度问题；修正尺度后可视化也应该更接近 GT。

### 图 4：最终三数据集量化结果柱状图

建议画 3 个小图，分别是 AbsRel、RMSE、delta1。每个小图中按数据集分组，对比 baseline 和 ours。

作用：比纯表格更直观，老师一眼能看到三个数据集上指标都是朝正确方向变化。

## 可视化图片是否太少

如果目前只放了 alley 这个 panorama-to-3D 示例，那么对于“最终成功结果”来说确实偏少。因为 alley 示例主要说明 DreamScene360 流程，不足以支撑三数据集量化结果。

建议最低配置如下：

| 类型 | 数量 | 是否必要 |
| --- | ---: | --- |
| 方法流程图 | 1 张 | 必要 |
| 三数据集深度对比图 | 3 张，每个数据集 1 张 | 必要 |
| Deep360 raw vs fixed-scale 修正图 | 1 张 | 建议放 |
| 指标柱状图 | 1 张 | 建议放 |

也就是说，实验部分至少应该有 5 张左右高信息量图。如果只放 1-2 张 alley 流程图，会显得可视化支撑不足；但也不需要放几十张，只要每张图都对应一个实验结论即可。

## 实验部分总结写法

综合量化结果和可视化结果可以得出：perspective-based PanoGeo fusion 对全景深度估计是有效的。对于 Matterport3D 和 Stanford2D3D，direct-guided blend 能够在保持 direct depth 全局稳定性的同时，引入 perspective fusion 的局部几何信息，因此三项指标均优于 direct baseline。对于 Deep360，raw metric scale 下结果失败主要来自尺度不匹配；经过 fixed-scale calibration 后，PanoGeo fusion 同样显著优于 direct baseline。

因此，本实验说明该方法的有效性依赖两个关键条件：第一，输入数据必须是真正的 equirectangular panorama；第二，单目深度预测的尺度需要合理校准。在满足这两个条件后，perspective-based fusion 能够稳定改善全景深度估计结果。
