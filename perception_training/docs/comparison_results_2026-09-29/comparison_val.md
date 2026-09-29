| Model | Params (M) | mIoU | ΔmIoU | Haz. mIoU | water rec. | mud rec. | puddle rec. | person rec. | val loss | test loss | GPU ms fp32 | GPU FPS fp16 | CPU ONNX ms | Infer MB | Train MB | Best ep. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| proxy_seed123_ep8 | 5.62 | 0.434 | — | 0.494 | n/a | 0.799 | 0.866 | 0.855 | 0.6444 | 1.5358 | 13.1 | 96 | 154 | 108 | 2744 | n/a |

**Per-class IoU (val)**

| class | proxy_seed123_ep8 |
|---|---|
| dirt | 0.000 |
| grass | 0.783 |
| tree | 0.762 |
| pole | 0.000 |
| water | 0.000 |
| sky | 0.956 |
| vehicle | 0.000 |
| object | 0.000 |
| asphalt | 0.000 |
| building | 0.000 |
| log | 0.341 |
| person | 0.384 |
| fence | 0.000 |
| bush | 0.557 |
| concrete | 0.689 |
| barrier | 0.562 |
| puddle | 0.418 |
| mud | 0.547 |
| rubble | 0.517 |

**Per-class recall (val)**

| class | proxy_seed123_ep8 |
|---|---|
| dirt | n/a |
| grass | 0.864 |
| tree | 0.941 |
| pole | 0.000 |
| water | n/a |
| sky | 0.973 |
| vehicle | 0.000 |
| object | 0.000 |
| asphalt | 0.000 |
| building | n/a |
| log | 0.849 |
| person | 0.855 |
| fence | n/a |
| bush | 0.653 |
| concrete | 0.865 |
| barrier | 0.945 |
| puddle | 0.866 |
| mud | 0.799 |
| rubble | 0.820 |
