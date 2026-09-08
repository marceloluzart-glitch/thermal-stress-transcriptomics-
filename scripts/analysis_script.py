import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
import statsmodels.api as sm

# Create output directories
os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)

file_path = "/content/drive/MyDrive/ML  BioInformática/BIOMARCADOR DE ESTRESS TERMICO /matrix_final_CPM_symbols_clean.csv"
df = pd.read_csv(file_path).set_index("symbol")

plt.figure(figsize=(7, 5))
sns.histplot(np.log1p(df.values.flatten()), bins=100, kde=True)
plt.title("Expression Value Distribution (log(CPM+1))")
plt.xlabel("log(CPM+1)")
plt.ylabel("Frequency")
plt.close()

meta = pd.DataFrame({
    "sample": df.columns,
    "cell": ["HEK293T" if "HEK293T" in s else "U2OS" for s in df.columns],
    "condition": [
        "HS" if "HS" in s else ("CTRL" if "CTRL" in s else "NA")
        for s in df.columns
    ],
}).set_index("sample")

X = np.log1p(df.T)
pca = PCA(n_components=2)
pc = pca.fit_transform(X)

pc_df = pd.DataFrame(pc, index=X.index, columns=["PC1", "PC2"])
pc_df = pc_df.join(meta)

plt.figure(figsize=(6, 6))
sns.scatterplot(data=pc_df, x="PC1", y="PC2", hue="cell", style="condition", s=120)
plt.title("Sample PCA (colors=cells, shapes=condition)")
plt.axhline(0, ls="--", c="grey")
plt.axvline(0, ls="--", c="grey")
plt.close()

link = linkage(X, method="ward")
plt.figure(figsize=(10, 5))
dendrogram(link, labels=X.index, leaf_rotation=90, leaf_font_size=10)
plt.title("Sample Hierarchical Clustering Dendrogram")
plt.ylabel("Distance")
plt.close()

variances = df.var(axis=1)
top_genes = variances.sort_values(ascending=False).head(50).index
df_top = df.loc[top_genes]

lut_cell = {"HEK293T": "#1f77b4", "U2OS": "#ff7f0e"}
lut_cond = {"HS": "#2ca02c", "CTRL": "#d62728", "NA": "#7f7f7f"}

col_colors = pd.DataFrame({
    "cell": meta["cell"].map(lut_cell),
    "condition": meta["condition"].map(lut_cond),
}, index=meta.index)

sns.clustermap(
    np.log1p(df_top),
    cmap="viridis",
    col_colors=col_colors,
    figsize=(12, 10),
    xticklabels=True,
    yticklabels=True,
)
plt.title("Heatmap - Top 50 Most Variable Genes", pad=80)
plt.close()

LOG2FC_THRESHOLD = 1.0
Q_VALUE_THRESHOLD = 0.05

meta_aed = pd.DataFrame({
    "sample_id": df.columns,
    "cell": [c.split("_")[0] for c in df.columns],
    "condition": [
        "HS" if "HS" in c else ("CTRL" if "CTRL" in c else "NA")
        for c in df.columns
    ],
})
meta_aed.loc[~meta_aed["condition"].isin(["HS", "CTRL"]), "condition"] = (
    "U2OS_NA"
)
meta_aed["group"] = meta_aed["cell"] + "_" + meta_aed["condition"]

samples_hek = meta_aed[meta_aed["cell"] == "HEK293T"]["sample_id"]
df_hek = df[samples_hek]
df_log2 = np.log2(df_hek + 1)

hs_samples = meta_aed[
    (meta_aed["cell"] == "HEK293T") & (meta_aed["condition"] == "HS")
]["sample_id"].tolist()
ctrl_samples = meta_aed[
    (meta_aed["cell"] == "HEK293T") & (meta_aed["condition"] == "CTRL")
]["sample_id"].tolist()

results_list = []
for gene in df_log2.index:
  exp_hs = df_log2.loc[gene, hs_samples].values
  exp_ctrl = df_log2.loc[gene, ctrl_samples].values
  log2fc = np.mean(exp_hs) - np.mean(exp_ctrl)
  try:
    _, p_value = ttest_ind(exp_hs, exp_ctrl, equal_var=False)
  except ValueError:
    p_value = 1.0
  results_list.append({"symbol": gene, "log2FC": log2fc, "p_value": p_value})

df_aed = pd.DataFrame(results_list).set_index("symbol")
p_values = df_aed["p_value"].fillna(1).values
_, q_values, _, _ = sm.stats.multipletests(p_values, method="fdr_bh")
df_aed["q_value"] = q_values

up_data = {
    "symbol": [
        "FYN",
        "FBXO44",
        "SFSWAP",
        "SPA17",
        "UST",
        "GPN2-AS1",
        "PPP1R13L",
        "TIMM23B",
        "IER3-AS1",
        "PIK3CD-AS2",
    ],
    "log2FC": [
        1.000280,
        1.169928,
        1.186695,
        1.154701,
        1.155674,
        2.513489,
        1.700649,
        2.248597,
        1.642448,
        1.118019,
    ],
    "q_value": [
        0.002135,
        0.002400,
        0.004195,
        0.006055,
        0.007169,
        0.007697,
        0.009354,
        0.009651,
        0.010454,
        0.012430,
    ],
}
df_up = pd.DataFrame(up_data)
df_up["Regulacao"] = "Induced (UP)"

down_data = {
    "symbol": [
        "RNU6-181P",
        "RNU6-353P",
        "RNU6-322P",
        "RNU6-313P",
        "RNU6-29P",
        "RNU6-291P",
        "RNU6-285P",
        "RNU6-268P",
        "RNU6-264P",
        "RNU6-25P",
    ],
    "log2FC": [
        -1.851997,
        -1.097610,
        -2.582554,
        -1.063502,
        -1.550899,
        -1.111030,
        -1.214124,
        -1.111030,
        -2.560713,
        -1.070388,
    ],
    "q_value": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}
df_down = pd.DataFrame(down_data)
df_down["Regulacao"] = "Repressed (DOWN)"

df_combined = pd.concat([df_up, df_down]).reset_index(drop=True)
df_combined = df_combined.sort_values(by="log2FC", ascending=True)

colors = {"Induced (UP)": "red", "Repressed (DOWN)": "blue"}

plt.figure(figsize=(10, 8))
plt.barh(
    df_combined["symbol"],
    df_combined["log2FC"],
    color=[colors[r] for r in df_combined["Regulacao"]],
)
plt.axvline(0, color="gray", linestyle="--", linewidth=1)
plt.xlabel("log₂ (Fold Change)", fontsize=12)
plt.ylabel("Gene Symbol", fontsize=12)
plt.title(
    "Top 20 Differentially Expressed Genes (HS vs. CTRL) - HEK293T",
    fontsize=14,
    fontweight="bold",
)

legend_elements = [
    Line2D(
        [0],
        [0],
        color="w",
        marker="o",
        markerfacecolor="red",
        markersize=10,
        label="Induced (UP)",
    ),
    Line2D(
        [0],
        [0],
        color="w",
        marker="o",
        markerfacecolor="blue",
        markersize=10,
        label="Repressed (DOWN)",
    ),
]
plt.legend(handles=legend_elements, loc="lower right", frameon=True)
plt.tight_layout()
plt.savefig("results/figures/top_20_de_genes.png", dpi=300)
plt.close()

df_aed["q_value"] = df_aed["q_value"].replace(0, 1e-300)
df_aed["-log10(q_value)"] = -np.log10(df_aed["q_value"])
Y_LIMIT_CLIP = 100
df_aed["Y_plot"] = df_aed["-log10(q_value)"].clip(upper=Y_LIMIT_CLIP)
NEG_LOG10_Q_THRESHOLD = -np.log10(Q_VALUE_THRESHOLD)


def categorize_gene_final(row):
  if (row["log2FC"] >= LOG2FC_THRESHOLD) and (row["q_value"] < Q_VALUE_THRESHOLD):
    return "Induced (UP)"
  elif (
      row["log2FC"] <= -LOG2FC_THRESHOLD
  ) and (row["q_value"] < Q_VALUE_THRESHOLD):
    return "Repressed (DOWN)"
  else:
    return "Not Significant"


df_aed["Regulacao"] = df_aed.apply(categorize_gene_final, axis=1)

df_plot_ordered = pd.concat([
    df_aed[df_aed["Regulacao"] == "Not Significant"],
    df_aed[df_aed["Regulacao"] != "Not Significant"],
])

plt.style.use("default")
plt.figure(figsize=(9, 7))

palette = {
    "Not Significant": "#A9A9A9",
    "Repressed (DOWN)": "#1F78B4",
    "Induced (UP)": "#E31A1C",
}

sns.scatterplot(
    x="log2FC",
    y="Y_plot",
    data=df_plot_ordered,
    hue="Regulacao",
    palette=palette,
    alpha=0.9,
    s=8,
    linewidth=0,
    zorder=2,
    hue_order=["Not Significant", "Repressed (DOWN)", "Induced (UP)"],
)

plt.axhline(
    NEG_LOG10_Q_THRESHOLD,
    color="black",
    linestyle="--",
    linewidth=1.5,
    zorder=1,
)
plt.axvline(LOG2FC_THRESHOLD, color="black", linestyle=":", linewidth=1.0, zorder=1)
plt.axvline(-LOG2FC_THRESHOLD, color="black", linestyle=":", linewidth=1.0, zorder=1)

top_genes_to_annotate = (
    df_aed[df_aed["Regulacao"] != "Not Significant"]
    .sort_values(by="q_value")
    .head(6)
)
max_y_value = df_aed["Y_plot"].max()

for symbol, row in top_genes_to_annotate.iterrows():
  if row["Y_plot"] >= max_y_value * 0.99:
    ha_align = "right" if row["log2FC"] < 0 else "left"
    xy_offset = (-10, 0) if row["log2FC"] < 0 else (10, 0)
    plt.annotate(
        symbol,
        (row["log2FC"], row["Y_plot"]),
        xytext=xy_offset,
        textcoords="offset points",
        fontsize=9,
        color=palette[row["Regulacao"]],
        ha=ha_align,
        fontweight="bold",
    )

plt.title(
    "Gene Expression Differences in Response to Thermal Stress",
    fontsize=14,
    fontweight="bold",
)
plt.xlabel("Log₂ (Fold Change: HS vs. CTRL)", fontsize=12)
plt.ylabel("-Log₁₀ (q-value)", fontsize=12)
plt.xlim(df_aed["log2FC"].min() - 0.5, df_aed["log2FC"].max() + 0.5)
plt.ylim(bottom=0, top=Y_LIMIT_CLIP * 1.05)
plt.legend(title="Regulation", loc="upper right", fontsize=10)

plt.text(
    x=df_aed["log2FC"].min() + 0.1,
    y=NEG_LOG10_Q_THRESHOLD + 0.5,
    s=f"Cutoff: q < {Q_VALUE_THRESHOLD}",
    fontsize=9,
    color="black",
)

sns.despine(trim=True)
plt.tight_layout()
plt.savefig("results/figures/volcano_plot_thermal_stress.png", dpi=300)
plt.close()
