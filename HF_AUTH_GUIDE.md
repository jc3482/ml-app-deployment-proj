# Hugging Face 认证指南

## 问题
Hugging Face 不再支持密码认证，需要使用 Access Token 或 SSH。

## 解决方案（选择一种）

### 方法 1: 使用 Access Token（最简单）

1. **获取 Token:**
   ```
   访问: https://huggingface.co/settings/tokens
   点击 "New token"
   名称: smartpantry-deploy
   权限: Write
   复制 token (格式: hf_xxxxxxxxxxxxx)
   ```

2. **更新 remote URL:**
   ```bash
   git remote set-url hf https://qqmian0820:YOUR_TOKEN@huggingface.co/spaces/qqmian0820/smartpantryy
   ```
   将 `YOUR_TOKEN` 替换为你复制的 token

3. **推送:**
   ```bash
   git push hf main
   git lfs push hf main
   ```

### 方法 2: 使用 Hugging Face CLI（推荐）

```bash
# 安装 CLI
pip install huggingface_hub

# 登录（会提示输入 token）
huggingface-cli login

# 然后正常推送
git push hf main
git lfs push hf main
```

### 方法 3: 使用 SSH

1. **添加 SSH key:**
   - 访问: https://huggingface.co/settings/keys
   - 添加你的 SSH 公钥

2. **更新 remote:**
   ```bash
   git remote set-url hf git@hf.co:spaces/qqmian0820/smartpantryy
   ```

3. **推送:**
   ```bash
   git push hf main
   git lfs push hf main
   ```
