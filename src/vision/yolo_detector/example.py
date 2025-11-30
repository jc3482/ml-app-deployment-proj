"""
智能冰箱食材检测 - 示例代码
"""

from food_detector import FoodDetector

def main():
    # 初始化检测器
    print("🧊 初始化食材检测器...")
    detector = FoodDetector('best.pt', conf_threshold=0.85)
    print("✓ 检测器就绪\n")
    
    # 示例图片路径（替换为你的图片）
    image_path = 'test_image.jpg'
    
    # 方法1: 简单列表
    print("=" * 50)
    print("方法 1: 简单列表（去重）")
    print("=" * 50)
    result = detector.detect(image_path)
    print(f"检测结果: {result}\n")
    
    # 方法2: 带置信度
    print("=" * 50)
    print("方法 2: 带置信度详情")
    print("=" * 50)
    result = detector.detect_with_confidence(image_path)
    if result:
        for name, conf in result:
            print(f"  • {name}: {conf:.1%}")
    else:
        print("  未检测到食材")
    print()
    
    # 方法3: 带数量
    print("=" * 50)
    print("方法 3: 带数量统计")
    print("=" * 50)
    result = detector.detect_with_count(image_path)
    if result:
        for name, count in sorted(result.items()):
            print(f"  • {name}: {count} 个")
    else:
        print("  未检测到食材")
    print()
    
    print("✅ 完成！")


if __name__ == "__main__":
    main()
