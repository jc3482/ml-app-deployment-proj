# Notebooks

This directory contains Jupyter notebooks for experimentation and analysis.

## Notebooks

### 01_data_exploration.ipynb
- Explore Food-101 and Recipe1M+ datasets
- Visualize data distributions
- Analyze ingredient frequencies
- Understand dataset characteristics

### 02_model_training.ipynb
- Train YOLOv8 for ingredient detection
- Fine-tune on Food-101 dataset
- Evaluate model performance
- Export trained models

### 03_recipe_retrieval.ipynb
- Build FAISS index for recipe search
- Generate recipe embeddings
- Test retrieval performance
- Evaluate ranking metrics

## Usage

```bash
# Install Jupyter
pip install jupyter notebook ipywidgets

# Start Jupyter server
jupyter notebook

# Or use JupyterLab
pip install jupyterlab
jupyter lab
```

## Best Practices

1. **Clear outputs before committing**: `jupyter nbconvert --clear-output --inplace *.ipynb`
2. **Use version control**: Keep notebooks in git but clear outputs
3. **Document experiments**: Add markdown cells explaining your approach
4. **Save results**: Export plots and metrics to files
5. **Modular code**: Move reusable code to `src/` modules

