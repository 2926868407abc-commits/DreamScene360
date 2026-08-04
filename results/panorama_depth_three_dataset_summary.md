# Three-Dataset Panorama Depth Summary

Evaluation protocol: scale-calibrated comparison. Lower AbsRel/RMSE is better; higher delta1 is better.

| Dataset | Baseline Method | Baseline AbsRel | Baseline RMSE | Baseline delta1 | Ours / Fusion Method | Ours AbsRel | Ours RMSE | Ours delta1 | Delta AbsRel | Delta RMSE | Delta delta1 | Num Images | Mean Metrics Better |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| Matterport3D | DA3-Direct-FixedScale-Calib10 | 0.4456 | 0.8291 | 0.4503 | DA3-DirectGuided-PanoGeo-P60-I300-Alpha04 | 0.3680 | 0.7376 | 0.5259 | -0.0775 | -0.0915 | +0.0755 | 133 | True |
| Stanford2D3D | DA3-Direct-FixedScale-Calib10 | 0.2537 | 0.7103 | 0.6286 | DA3-DirectGuided-PanoGeo-P60-I300-Alpha04 | 0.2136 | 0.6794 | 0.6397 | -0.0401 | -0.0309 | +0.0112 | 133 | True |
| Deep360 | DA3-Direct-Deep360-FixedScale-Calib10 | 2.2448 | 18.5921 | 0.1528 | DreamScene360-PanoGeo-DA3-P24-I100-FixedScale-Calib10 | 0.6195 | 12.2246 | 0.3002 | -1.6253 | -6.3674 | +0.1474 | 133 | True |

Notes:

- Matterport3D and Stanford2D3D use the P60-I300 direct-guided PanoGeo blend result.
- Deep360 uses the scale-calibrated P24-I100 PanoGeo result, because this is the currently validated effective Deep360 run.
- Delta is computed as Ours minus Baseline. Therefore, negative Delta AbsRel/RMSE and positive Delta delta1 indicate improvement.
