# 训练指南 (Training Guide)

本指南将帮助你进行小规模测试和完整训练。

## 步骤 1: 小规模测试 (必须先做！)

在开始完整训练之前，强烈建议先运行小规模测试来验证所有组件是否正常工作。

### 运行测试脚本

```bash
# 在项目根目录下运行
python src/vision/test_setup.py
```

### 测试内容

测试脚本会依次检查：

1. **数据加载** - 验证能否正确加载YOLO格式数据
2. **模型前向传播** - 验证模型能否正常处理输入
3. **损失计算** - 验证损失函数是否正常工作
4. **反向传播** - 验证梯度计算是否正常
5. **小训练步骤** - 验证完整的训练步骤（2个iteration）

### 预期输出

如果一切正常，你会看到：

```
============================================================
SMALL-SCALE TESTING - Verifying Setup
============================================================

============================================================
Test 1: Data Loading
============================================================
✓ Dataset loaded successfully
  Total samples: 2895
✓ Sample loaded successfully
  Image shape: torch.Size([3, 640, 640])
  Number of boxes: 3
...

============================================================
✅ ALL TESTS PASSED!
============================================================

Your setup is ready for full training.
You can now run: python src/vision/train_full.py
```

### 如果测试失败

- **数据加载失败**: 检查 `data/fridge_photos/data.yaml` 中的路径是否正确
- **模型创建失败**: 检查是否安装了所有依赖（torch, torchvision）
- **CUDA错误**: 如果没有GPU，代码会自动使用CPU，但训练会很慢

---

## 步骤 2: 完整训练

测试通过后，可以开始完整训练。

### 运行训练脚本

```bash
# 在项目根目录下运行
python src/vision/train_full.py
```

### 训练配置

默认配置（可在 `train_full.py` 中修改）：

- **Epochs**: 100
- **Batch size**: 8 (GPU) / 2 (CPU)
- **Learning rate**: 0.001
- **Image size**: 640x640
- **训练数据**: 2,895 张图片
- **验证数据**: 103 张图片

### 训练过程监控

#### 1. 控制台输出

训练过程中会显示：
- 每个epoch的训练损失
- 验证损失和mAP
- 学习率变化

#### 2. TensorBoard可视化

在另一个终端运行：

```bash
tensorboard --logdir runs/detection_*
```

然后在浏览器打开 `http://localhost:6006` 查看：
- 训练/验证损失曲线
- 学习率曲线
- 各个损失组件的变化

### 训练时间估算

- **GPU (NVIDIA)**: 约 4-8 小时（100 epochs）
- **Apple Silicon (MPS)**: 约 8-12 小时
- **CPU**: 约 24-48 小时（不推荐）

### 模型保存

训练过程中会自动保存：
- **`models/checkpoints/latest.pth`**: 最新检查点（每10个epoch）
- **`models/checkpoints/best.pth`**: 最佳模型（验证mAP最高时）

### 中断和恢复训练

如果训练中断，可以修改代码恢复训练：

```python
# 在 train_full.py 的 main() 函数末尾，trainer.train() 之前添加：
checkpoint_path = Path("models/checkpoints/latest.pth")
if checkpoint_path.exists():
    trainer.load_checkpoint(checkpoint_path)
    logger.info(f"Resuming from checkpoint: {checkpoint_path}")

trainer.train()
```

---

## 步骤 3: 调整超参数（可选）

如果训练效果不理想，可以调整以下参数：

### 在 `train_full.py` 中修改配置

```python
config = {
    "training": {
        "epochs": 100,           # 增加训练轮数
        "batch_size": 8,         # 根据GPU内存调整（8GB GPU用4，16GB用8）
        "learning_rate": 0.001,  # 如果损失不下降，尝试 0.0001
        "cls_weight": 1.0,       # 分类损失权重
        "reg_weight": 5.0,       # 回归损失权重（通常比分类大）
        "obj_weight": 1.0,       # 置信度损失权重
    },
}
```

### 常见调整策略

1. **损失不下降**:
   - 降低学习率到 0.0001
   - 增加训练轮数

2. **过拟合**:
   - 增加数据增强
   - 增加 weight_decay 到 0.001

3. **显存不足**:
   - 减小 batch_size 到 4 或 2
   - 减小 img_size 到 512

---

## 使用训练好的模型

训练完成后，加载模型进行推理：

```python
from src.vision.detector import CustomDetector
import torch

# 加载模型
model = CustomDetector(num_classes=30)
checkpoint = torch.load("models/checkpoints/best.pth")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# 推理（需要实现完整的后处理）
# 目前 predict() 方法还需要完善
```

---

## 常见问题

### Q: 训练时显存不足怎么办？

A: 减小 batch_size：
```python
config["training"]["batch_size"] = 4  # 或 2
```

### Q: 训练很慢怎么办？

A: 
- 确保使用GPU（检查 `device` 输出）
- 减小 `num_workers` 如果CPU核心数少
- 考虑先用更少的epochs测试（如10个epochs）

### Q: 如何只训练几个epochs测试？

A: 修改配置：
```python
config["training"]["epochs"] = 5  # 只训练5个epochs
```

### Q: 数据路径错误怎么办？

A: 检查 `data/fridge_photos/data.yaml`：
```yaml
train: data/fridge_photos/train/images  # 相对路径
val: data/fridge_photos/valid/images
```

确保路径是相对于项目根目录的。

---

## 下一步

训练完成后：
1. 评估模型性能（在测试集上）
2. 实现完整的后处理（NMS、坐标转换）
3. 集成到主应用 (`app/main.py`)
4. 与YOLOv8对比性能

