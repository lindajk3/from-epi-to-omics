import pandas as pd
import numpy as np
from scipy.stats import zscore, mannwhitneyu
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns

# load real human blood metabolomics data (MTBLS265)
# study: individual variability in human blood metabolites - age-related differences
# 30 subjects, whole blood
df = pd.read_csv('m_MTBLS265_POS_mass_spectrometry_v2_maf.tsv', sep='\t')

print("Shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

# extract metabolite names from the dataset
metabolite_names = df['metabolite_identification']

# separate the sample columns from the rest
sample_cols = [col for col in df.columns if 'Person' in col]

# filter to participant measurements only
data = df[sample_cols].copy()
data.index = metabolite_names

# group participants by age category
youth_cols = [col for col in sample_cols if 'youth' in col]
elder_cols = [col for col in sample_cols if 'elder' in col]

print("Youth samples:", len(youth_cols))
print("Elder samples:", len(elder_cols))
print("\nData ready for analysis:")
print(data.head())

# data transformation and standardisation
data_log = np.log1p(data)
data_normalised = data_log.apply(zscore)

print("\nAfter normalisation:")
print(data_normalised.head())
print("\nMean should be ~0:", data_normalised.mean().mean().round(4))
print("Std should be ~1:", data_normalised.std().mean().round(4))

# PCA to visualise group separation
data_T = data_normalised.T
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data_T)

pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
pca_df['group'] = ['Youth' if 'youth' in col else 'Elder' for col in data_T.index]

colours = {'Youth': '#2166ac', 'Elder': '#d6604d'}
fig, ax = plt.subplots(figsize=(9, 7))

for group, colour in colours.items():
    subset = pca_df[pca_df['group'] == group]
    ax.scatter(subset['PC1'], subset['PC2'], label=group,
               color=colour, s=100, alpha=0.85, edgecolors='white', linewidths=0.5)

    mean_x, mean_y = subset['PC1'].mean(), subset['PC2'].mean()
    cov = np.cov(subset['PC1'], subset['PC2'])
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(*eigenvectors[:, 1][::-1]))
    width, height = 2 * 2 * np.sqrt(eigenvalues)
    ellipse = Ellipse((mean_x, mean_y), width, height, angle=angle,
                      color=colour, alpha=0.15)
    ax.add_patch(ellipse)

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
ax.set_title('PCA of Blood Metabolite Profiles\nYouth vs Elder (MTBLS265)', fontsize=13, fontweight='bold')
ax.legend(title='Age Group', fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('pca_plot.png', dpi=200, bbox_inches='tight')
plt.show()
print("PCA plot saved!")

# compare metabolite levels between age groups
results = []

for metabolite in data_normalised.index:
    youth_values = data_normalised.loc[metabolite, youth_cols]
    elder_values = data_normalised.loc[metabolite, elder_cols]
    
    stat, pval = mannwhitneyu(youth_values, elder_values, alternative='two-sided')
    
    mean_youth = data.loc[metabolite, youth_cols].mean()
    mean_elder = data.loc[metabolite, elder_cols].mean()
    fold_change = np.log2(mean_elder / mean_youth)
    
    results.append({
        'metabolite': metabolite,
        'p_value': pval,
        'log2_fold_change': fold_change
    })

results_df = pd.DataFrame(results)

# adjust p-values for multiple testing (FDR correction)
results_df['p_adj'] = multipletests(results_df['p_value'], method='fdr_bh')[1]
results_df = results_df.sort_values('p_adj')

print("\nTop 10 metabolites different between youth and elder:")
print(results_df.head(10))

# volcano plot
fig, ax = plt.subplots(figsize=(9, 7))

colours_v = []
for _, row in results_df.iterrows():
    if row['p_adj'] < 0.05 and abs(row['log2_fold_change']) > 0.5:
        colours_v.append('tomato')
    else:
        colours_v.append('lightgrey')

ax.scatter(results_df['log2_fold_change'],
           -np.log10(results_df['p_value']),
           c=colours_v, s=80, alpha=0.8, edgecolors='white', linewidths=0.5)

sig = results_df[results_df['p_adj'] < 0.05]
for _, row in sig.iterrows():
    ax.annotate(row['metabolite'],
                (row['log2_fold_change'], -np.log10(row['p_value'])),
                fontsize=8, ha='center', va='bottom',
                xytext=(0, 5), textcoords='offset points')

ax.axhline(-np.log10(0.05), color='grey', linestyle='--', linewidth=0.8)
ax.axvline(0.5, color='grey', linestyle='--', linewidth=0.8)
ax.axvline(-0.5, color='grey', linestyle='--', linewidth=0.8)

ax.set_xlabel('Log2 Fold Change (Elder / Youth)', fontsize=12)
ax.set_ylabel('-Log10 (p-value)', fontsize=12)
ax.set_title('Volcano Plot: Blood Metabolites\nYouth vs Elder (MTBLS265)', fontsize=13, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('volcano_plot.png', dpi=200, bbox_inches='tight')
plt.show()
print("Volcano plot saved!")

# heatmap of significant metabolites
top_metabolites = results_df[results_df['p_adj'] < 0.05]['metabolite'].tolist()
heatmap_data = data_normalised.loc[top_metabolites]

sorted_cols = youth_cols + elder_cols
heatmap_data = heatmap_data[sorted_cols]

col_labels = ['Y'+str(i+1) if 'youth' in col else 'E'+str(i+1)
              for i, col in enumerate(heatmap_data.columns)]
heatmap_data.columns = col_labels

fig, ax = plt.subplots(figsize=(12, 5))
sns.heatmap(heatmap_data,
            cmap='RdBu_r',
            center=0,
            ax=ax,
            linewidths=0.3,
            cbar_kws={'label': 'Z-score'})

ax.set_title('Heatmap of Significant Metabolites\nYouth vs Elder (MTBLS265)', fontsize=13, fontweight='bold')
ax.set_xlabel('Samples (Y=Youth, E=Elder)', fontsize=11)
ax.set_ylabel('Metabolite', fontsize=11)

plt.tight_layout()
plt.savefig('heatmap.png', dpi=200, bbox_inches='tight')
plt.show()
print("Heatmap saved!")
# box plots for top significant metabolites
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, metabolite in enumerate(top_metabolites):
    youth_vals = data.loc[metabolite, youth_cols].values
    elder_vals = data.loc[metabolite, elder_cols].values
    
    ax = axes[i]
    ax.boxplot([youth_vals, elder_vals], 
               labels=['Youth', 'Elder'],
               patch_artist=True,
               boxprops=dict(facecolor='lightblue', color='steelblue'),
               medianprops=dict(color='darkblue', linewidth=2))
    
    # add individual data points
    ax.scatter([1]*len(youth_vals), youth_vals, color='#2166ac', alpha=0.6, zorder=3, s=40)
    ax.scatter([2]*len(elder_vals), elder_vals, color='#d6604d', alpha=0.6, zorder=3, s=40)
    
    p_val = results_df[results_df['metabolite'] == metabolite]['p_adj'].values[0]
    ax.set_title(f'{metabolite}\n(p_adj = {p_val:.4f})', fontsize=11, fontweight='bold')
    ax.set_ylabel('Abundance', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle('Top Significant Metabolites: Youth vs Elder (MTBLS265)', 
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('boxplots.png', dpi=200, bbox_inches='tight')
plt.show()
print("Box plots saved!")
# pathway analysis - map significant metabolites to biological pathways
pathways = {
    'Amino acid metabolism': ['Citrulline', 'N-Acetyl-arginine', 'Tyrosine', 
                               'Trimethyl-tyrosine', 'S-Adenosyl-homocysteine'],
    'Antioxidant/Anti-aging': ['Carnosine', 'Acetylcarnosine'],
    'Nucleotide metabolism': ['ADP', 'AMP', 'ATP', 'GTP', 'CTP', 'NADP+'],
    'Energy metabolism': ['Pyruvate', 'Succinate', 'Citrate', 'Lactate'],
    'Methylation cycle': ['S-Adenosyl-homocysteine']
}

# count how many significant metabolites fall into each pathway
sig_metabolites = results_df[results_df['p_adj'] < 0.05]['metabolite'].tolist()

pathway_counts = {}
for pathway, members in pathways.items():
    hits = [m for m in sig_metabolites if m in members]
    pathway_counts[pathway] = len(hits)

pathway_df = pd.DataFrame(list(pathway_counts.items()), 
                           columns=['Pathway', 'Significant Metabolites'])
pathway_df = pathway_df[pathway_df['Significant Metabolites'] > 0]
pathway_df = pathway_df.sort_values('Significant Metabolites', ascending=True)

# plot
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(pathway_df['Pathway'], pathway_df['Significant Metabolites'],
               color='steelblue', alpha=0.85)

ax.set_xlabel('Number of Significant Metabolites', fontsize=12)
ax.set_title('Enriched Biological Pathways\nYouth vs Elder (MTBLS265)', 
             fontsize=13, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for bar, val in zip(bars, pathway_df['Significant Metabolites']):
    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
            str(val), va='center', fontsize=11)

plt.tight_layout()
plt.savefig('pathway_analysis.png', dpi=200, bbox_inches='tight')
plt.show()
print("Pathway analysis saved!")
# correlation heatmap between metabolites
corr_matrix = data_normalised.T.corr()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix,
            cmap='RdBu_r',
            center=0,
            vmin=-1, vmax=1,
            ax=ax,
            linewidths=0.3,
            xticklabels=True,
            yticklabels=True,
            cbar_kws={'label': 'Pearson Correlation'})

ax.set_title('Metabolite Correlation Matrix\nBlood Metabolites (MTBLS265)',
             fontsize=13, fontweight='bold')
ax.tick_params(axis='x', labelsize=7, rotation=90)
ax.tick_params(axis='y', labelsize=7)

plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=200, bbox_inches='tight')
plt.show()
print("Correlation heatmap saved!")
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

# hierarchical clustering of samples by metabolite profile
fig, ax = plt.subplots(figsize=(12, 5))

linked = linkage(data_normalised.T, method='ward')

# create simple labels
simple_labels = []
youth_count = 0
elder_count = 0
for col in data_normalised.columns:
    if 'youth' in col:
        youth_count += 1
        simple_labels.append(f'Y{youth_count}')
    else:
        elder_count += 1
        simple_labels.append(f'E{elder_count}')

dend = dendrogram(linked,
                  labels=simple_labels,
                  ax=ax,
                  leaf_rotation=90)

# colour leaf labels
for lbl in ax.get_xticklabels():
    if lbl.get_text().startswith('Y'):
        lbl.set_color('#2166ac')
    else:
        lbl.set_color('#d6604d')

ax.set_title('Hierarchical Clustering of Blood Metabolite Profiles\nYouth vs Elder (MTBLS265)',
             fontsize=13, fontweight='bold')
ax.set_ylabel('Distance', fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('clustering.png', dpi=200, bbox_inches='tight')
plt.show()
print("Clustering plot saved!")