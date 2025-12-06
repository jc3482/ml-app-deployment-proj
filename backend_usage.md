# Backend & Recipe Matching System 使用指南

这份文档详细说明了后端（Backend）和食谱匹配系统（Recipe Matching System）的工作流程。请参考此逻辑进行前端 App 开发和部署配置。

## 1. 数据源 (Data Source)

系统的原始数据位于：
`data/recipes/recipe_dataset_raw.csv`

## 2. 数据处理流程 (Data Processing Pipeline)

在运行推荐系统之前，需要对原始数据进行两步预处理：标准化（Normalization）和本体处理（Ontology Processing）。

### 步骤 1: 数据标准化 (Recipe Dataset Normalization)

这个步骤读取原始 CSV 数据，解析并标准化食材名称。

**运行命令:**
```bash
python -m recipe_matcher.bin.main normalize
```

**执行逻辑:**
1. 加载 `data/recipes/recipe_dataset_raw.csv` (约 13501 条食谱)。
2. 初始化 `IngredientNormalizer`。
3. 解析并清洗食材列表。
4. 移除没有有效食材的食谱。
5. **输出文件:**
   - `data/normalized_recipes.json`
   - `data/normalized_recipes.csv`

### 步骤 2: 本体处理 (Ontology Processing)

这个步骤在标准化的基础上，进一步将食材映射到规范的本体（Ontology）形式，以便更准确地匹配。

**运行命令:**
```bash
python -m recipe_matcher.bin.main ontology
```

**执行逻辑:**
1. 加载上一步生成的 `data/normalized_recipes.json`。
2. 初始化 `IngredientOntology`。
3. 应用本体规范化（Canonicalization）。
4. **输出文件:**
   - `data/ontology_recipes.json`
   - `data/ontology_recipes.csv`
   - `data/ontology_recipes.pkl` (用于后端快速加载)

## 3. 后端推荐逻辑测试 (Backend Testing & Logic)

处理完数据后，可以使用 `src/backend/recipe_recommender.py` 脚本来测试完整的后端逻辑。这个脚本展示了从图片输入到最终食谱推荐的完整链路。

**运行命令:**
```bash
export PYTHONPATH=$PYTHONPATH:.:./recipe_matching_system && python src/backend/recipe_recommender.py
```

**完整后端逻辑 (Pipeline):**

1. **初始化 (Initialization):**
   - 加载 `data/ontology_recipes.pkl` 数据集。
   - 初始化 `RecipePipeline` (采用 Retrieve & Rank 架构)。

2. **图像输入 (Input Image):**
   - 系统接收一张冰箱食材的图片（例如 `data/fridge_photos/test/images/...`）。

3. **食材检测 (Object Detection):**
   - 使用 YOLO 模型对图片进行推理。
   - 识别出原始食材列表 (Raw YOLO detections)，例如 `['butter', 'onion', 'eggs', ...]`。

4. **食材扩展 (Ingredient Expansion):**
   - 自动添加常见的家中常备食材 (Pantry items)，例如盐、油、糖等。
   - 生成最终的用户食材列表 (User ingredients)。

5. **食谱搜索与匹配 (Search & Match):**
   - 在 13489 条处理过的食谱中进行搜索。
   - **检索 (Retrieve):** 首先筛选出 300 个候选食谱。
   - **排序 (Rank):** 根据食材匹配度对候选食谱进行打分和排序。

6. **输出结果 (Output):**
   - 返回得分最高的 5 个食谱 (Top Matches)。
   - 每个推荐包含：
     - 食谱名称 (Recipe Name)
     - 匹配得分 (Score)
     - 所需食材 (Ingredients)
     - 已有食材 (Matched Ingredients)
     - 缺失食材 (Missing Ingredients)

## 给前端/部署开发的备注

- **API 逻辑:** 前端 App 的后端接口应当复用 `src/backend/recipe_recommender.py` 中的 `RecipePipeline` 逻辑。
- **输入:** 用户上传的图片。
- **处理:** 调用模型识别 -> 补充常备食材 -> 匹配数据库。
- **输出:** JSON 格式的推荐食谱列表展示给用户。
- **依赖:** 部署时请确保包含 `recipe_matching_system` 目录以及生成的 `data/ontology_recipes.pkl` 文件。

