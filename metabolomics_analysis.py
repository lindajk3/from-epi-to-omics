import pandas as pd

# Load real human blood metabolomics data (MTBLS265)
# Study: Individual variability in human blood metabolites - age-related differences
# 30 subjects, whole blood, published in PNAS

df = pd.read_csv('m_MTBLS265_POS_mass_spectrometry_v2_maf.tsv', sep='\t')

print("Shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())
print("\nFirst few rows:")
print(df.head())
# Separate metabolite names and sample data
metabolite_names = df['metabolite_identification']

# Extract only the sample columns (youth and elder)
sample_cols = [col for col in df.columns if 'Person' in col]
data = df[sample_cols].copy()
data.index = metabolite_names

# Separate youth and elder
youth_cols = [col for col in sample_cols if 'youth' in col]
elder_cols = [col for col in sample_cols if 'elder' in col]

print("Youth samples:", len(youth_cols))
print("Elder samples:", len(elder_cols))
print("\nData ready for analysis:")
print(data.head())
import numpy as np
from scipy.stats import zscore

# Log transform first (common in metabolomics)
data_log = np.log1p(data)

# Then standardise (zscore)
data_normalised = data_log.apply(zscore)

print("\nAfter normalisation:")
print(data_normalised.head())
print("\nMean should be ~0:", data_normalised.mean().mean().round(4))
print("Std should be ~1:", data_normalised.std().mean().round(4))
import numpy as np
from scipy.stats import zscore

# Log transform first (common in metabolomics)
data_log = np.log1p(data)

# Then standardise (zscore)
data_normalised = data_log.apply(zscore)

print("\nAfter normalisation:")
print(data_normalised.head())
print("\nMean should be ~0:", data_normalised.mean().mean().round(4))
print("Std should be ~1:", data_normalised.std().mean().round(4))
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Transpose so rows = people, columns = metabolites
data_T = data_normalised.T

# PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data_T)

# Create a dataframe for plotting
pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
pca_df['group'] = ['Youth' if 'youth' in col else 'Elder' for col in data_T.index]

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

fig, ax = plt.subplots(figsize=(9, 7))

colours = {'Youth': '#2166ac', 'Elder': '#d6604d'}

for group, colour in colours.items():
    subset = pca_df[pca_df['group'] == group]
    ax.scatter(subset['PC1'], subset['PC2'], label=group,
               color=colour, s=100, alpha=0.85, edgecolors='white', linewidths=0.5)

    # Confidence ellipse
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