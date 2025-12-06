# 智能冰箱食材检测器

基于 YOLOv8 的高精度食材识别系统

## 性能指标

- **mAP50**: 95%
- **mAP50-95**: 60%
- **默认置信度阈值**: 85%

## 快速开始

### 1. 安装依赖
```bash
pip install ultralytics pillow
```

### 2. 使用检测器
```python
from food_detector import FoodDetector

# 初始化检测器
detector = FoodDetector('best.pt', conf_threshold=0.85)

# 检测食材
result = detector.detect('image.jpg')
print(result)
# 输出: ['Apple', 'Banana', 'Tomato']
```

## 使用方法

### 方法 1: 简单列表（去重）
```python
result = detector.detect('image.jpg')
# 返回: ['Apple', 'Banana', 'Tomato']
```

### 方法 2: 带置信度详情
```python
result = detector.detect_with_confidence('image.jpg')
# 返回: [('Apple', 0.95), ('Banana', 0.92), ('Tomato', 0.87)]

for name, confidence in result:
    print(f"{name}: {confidence:.1%}")
```

### 方法 3: 带数量统计
```python
result = detector.detect_with_count('image.jpg')
# 返回: {'Apple': 2, 'Banana': 1, 'Tomato': 3}

for name, count in result.items():
    print(f"{name}: {count}个")
```

### 自定义置信度阈值
```python
# 使用更低的阈值（检测更多物体，可能有误报）
result = detector.detect('image.jpg', conf_threshold=0.70)

# 使用更高的阈值（只检测高置信度物体）
result = detector.detect('image.jpg', conf_threshold=0.90)
```

## 文件说明

- `best.pt` - YOLOv8 训练好的模型权重文件
- `food_detector.py` - 检测器类封装
- `config.pkl` - 模型配置信息
- `README.md` - 本说明文档

## 示例代码
```python
from food_detector import FoodDetector
from PIL import Image
import matplotlib.pyplot as plt

# 初始化
detector = FoodDetector()

# 检测
ingredients = detector.detect('fridge_photo.jpg')

# 显示结果
print(f"检测到 {len(ingredients)} 种食材:")
for item in ingredients:
    print(f"  • {item}")

# 详细信息
details = detector.detect_with_confidence('fridge_photo.jpg')
print("\n详细信息:")
for name, conf in details:
    print(f"  • {name}: 置信度 {conf:.1%}")
```

## 注意事项

1. 需要 Python 3.8+
2. 建议在 GPU 环境下运行以获得更快速度
3. 支持的图片格式：JPG, PNG, JPEG
4. 默认置信度阈值为 85%，可根据需求调整

## 技术支持

如有问题，请检查：
1. ultralytics 是否正确安装
2. 模型文件 `best.pt` 是否存在
3. 图片路径是否正确

## License

MIT License
