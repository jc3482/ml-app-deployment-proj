#!/usr/bin/env python3
"""
简单测试脚本 - 测试 API 是否能正常启动和响应
"""
import requests
import time
import sys

API_URL = "http://localhost:8001"

def test_health():
    """测试健康检查端点"""
    print("🔍 测试 /health 端点...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        print(f"✅ 状态码: {response.status_code}")
        print(f"✅ 响应: {response.json()}")
        return True
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到 API，请确保 API 正在运行")
        print(f"   运行命令: uvicorn app.api_extended:app --host 0.0.0.0 --port 8001")
        return False
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False

def test_root():
    """测试根端点"""
    print("\n🔍 测试 / 端点...")
    try:
        response = requests.get(f"{API_URL}/", timeout=5)
        print(f"✅ 状态码: {response.status_code}")
        print(f"✅ 响应: {response.json()}")
        return True
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False

def test_api_root():
    """测试 API 根端点"""
    print("\n🔍 测试 /api 端点...")
    try:
        # 测试一个简单的 API 端点
        response = requests.get(f"{API_URL}/api/pantry/list", timeout=5)
        print(f"✅ 状态码: {response.status_code}")
        print(f"✅ 响应: {response.json()}")
        return True
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False

def main():
    print("=" * 50)
    print("SmartPantry API 测试")
    print("=" * 50)
    print(f"\n🌐 API URL: {API_URL}")
    print("\n⏳ 等待 API 启动...")
    time.sleep(2)
    
    results = []
    results.append(("健康检查", test_health()))
    results.append(("根端点", test_root()))
    results.append(("API 端点", test_api_root()))
    
    print("\n" + "=" * 50)
    print("测试结果总结:")
    print("=" * 50)
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name}: {status}")
    
    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n🎉 所有测试通过！API 运行正常。")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查 API 日志。")
        return 1

if __name__ == "__main__":
    sys.exit(main())

