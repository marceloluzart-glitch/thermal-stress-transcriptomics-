# Thermal Stress Transcriptomic Analysis

Pipeline for quality control, exploratory data analysis, and differential expression analysis (AED) of RNA-Seq data evaluating cellular responses to thermal stress (Heat Shock - HS vs. Control - CTRL) in HEK293T and U2OS cell lines.

## Pipeline Steps
1. **Data Preprocessing & QC:** Log-transformation, sample distribution checks, PCA, and hierarchical clustering.
2. **Exploratory Heatmap:** Visualization of the top 50 most variable genes.
3. **Differential Expression Analysis (DEA):** Welch's t-test combined with Benjamini-Hochberg FDR correction ($q$-value) for HEK293T cells.
4. **Visualizations:** 
   - Top 20 differentially expressed genes bar plot.
   - Publication-ready Volcano Plot with significance thresholds.

## Requirements & Installation
Install the required dependencies via pip:
```bash
pip install -r requirements.txt
