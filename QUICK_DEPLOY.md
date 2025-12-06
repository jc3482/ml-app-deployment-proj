# Quick Deployment Guide

## Docker 本地部署

```bash
# 1. 构建镜像
docker build -t smartpantry:latest .

# 2. 运行容器
docker-compose up -d

# 3. 查看日志
docker-compose logs -f

# 4. 访问应用
# API: http://localhost:8001
# 健康检查: http://localhost:8001/health
# API 文档: http://localhost:8001/docs
```

## Hugging Face Spaces 部署

### 步骤 1: 准备 Git LFS（用于大文件）

```bash
# 安装 Git LFS（如果还没安装）
git lfs install

# 追踪大文件
git lfs track "data/ontology_recipes.pkl"
git lfs track "data/normalized_recipes.pkl"
git add .gitattributes
git commit -m "Add Git LFS tracking"
```

### 步骤 2: 创建 Hugging Face Space

1. 访问 https://huggingface.co/spaces
2. 点击 "Create new Space"
3. 配置：
   - **Name**: `smartpantry`
   - **SDK**: `Docker`
   - **Hardware**: CPU Basic

### 步骤 3: 推送代码

```bash
# 添加 Hugging Face remote
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/smartpantry

# 推送代码
git push hf main

# 如果有大文件，使用 LFS 推送
git lfs push hf main
```

### 步骤 4: 等待构建完成

- 构建时间：约 10-15 分钟
- 在 Hugging Face Space 页面查看构建日志
- 构建完成后访问你的 Space URL

## 重要提示

1. **数据文件大小**：
   - `ontology_recipes.pkl`: 36MB（需要 Git LFS）
   - Hugging Face Spaces 限制：50GB per space

2. **端口配置**：
   - Hugging Face 会自动设置 `PORT` 环境变量
   - Dockerfile 使用 `${PORT:-8001}` 作为默认值

3. **健康检查**：
   - Hugging Face 会自动检查 `/health` 端点
   - 确保返回 200 OK

## 故障排除

- **构建失败**: 查看 Hugging Face 构建日志
- **模块找不到**: 检查 `recipe_matching_system/` 是否在仓库中
- **大文件上传失败**: 确保使用 `git lfs push`
