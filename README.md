##UHFusion
## 工程结构

```text
.
├── configs/pavia.json          # 数据、模型、训练和测试配置
├── tgrs_fusion/
│   ├── data/                   # MAT 数据集、任务构造、批处理
│   ├── models/                 # 双域专家、Swin、融合主模型
│   └── engine/                 # 训练、验证、测试、指标、断点
├── train.py                    # 训练入口
├── test.py                     # 四类任务测试入口
├── smoke_test.py               # 无数据结构检查
├── checkpoints/                # 归档权重
└── legacy/                     # 原始脚本，只作追溯
```


## 任务编号

- `0`：PAN
- `1`：从 GT-HSI 选取的四个波段
- `2`：由 SRF 投影得到的 MSI
- `3`：LR-HSI 第一主成分（缺失辅助模态情形）
