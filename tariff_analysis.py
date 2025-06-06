import matplotlib
matplotlib.use('Agg')  # Prevent GUI issues in headless environments

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle

# === Load and Filter Data ===
print("Loading CSV data...")
data = pd.read_csv("2020_SML.csv", index_col=0)
chn_usa_sectors = [s for s in data.index if s.startswith('CHN_') or s.startswith('USA_')]
data = data.loc[chn_usa_sectors, chn_usa_sectors]
common_sectors = data.index.intersection(data.columns)
data = data.loc[common_sectors, common_sectors]

# === Normalize to Technical Coefficient Matrix A ===
col_sums = data.sum(axis=0)
col_sums[col_sums.abs() < 1e-8] = 1e-8
A = data.divide(col_sums, axis=1)

# === Apply 25% Tariff on CHN_C26 (Multiplicative) ===
A_tariff = A.copy()
if 'CHN_C26' in A_tariff.columns:
    A_tariff.loc[:, 'CHN_C26'] *= 1.25

# === Leontief Inverse and Output Calculation ===
I = np.eye(len(A))
L_base = np.linalg.inv(I - A.values)
L_tariff = np.linalg.inv(I - A_tariff.values)
final_demand = np.ones((len(A), 1)) * 1e-6
output_base = L_base @ final_demand
output_tariff = L_tariff @ final_demand
abs_change = output_tariff - output_base
percent_change = (abs_change / output_base) * 100

# === Result DataFrame ===
results = pd.DataFrame({
    'Sector': A.columns,
    'Baseline': output_base.flatten(),
    'PostTariff': output_tariff.flatten(),
    'Abs_Change': abs_change.flatten(),
    'Pct_Change': percent_change.flatten()
})
results['Country'] = results['Sector'].str[:3]
results['SectorCode'] = results['Sector'].str[4:]

# === Load Sector Names from Excel ===
print("Loading sector names from Excel...")
xls = pd.ExcelFile("ReadMe_ICIO_small.xlsx")
df = xls.parse('Area_Activities')
sector_info = df.iloc[2:, [7, 8, 9]]
sector_info.columns = ['Old_Code', 'Code', 'Sector_Name']
sector_info = sector_info.dropna(subset=['Code', 'Sector_Name'])
sector_info['SectorShort'] = sector_info['Code'].str.extract(r'([A-Z0-9_]+)$')
sector_map = dict(zip(sector_info['SectorShort'], sector_info['Sector_Name']))
results['SectorName'] = results['SectorCode'].map(sector_map).fillna(results['SectorCode'])

# === Save Results ===
results.to_csv("tariff_results.csv", index=False)
print("✅ Saved: tariff_results.csv")

# === Bar Chart: Top 10 Most Negatively Affected Sectors ===
top10 = results.sort_values(by='Abs_Change').head(10)
labels = top10['SectorName'] + ' (' + top10['Country'] + ')'

plt.figure(figsize=(12, 7))
bars = plt.barh(labels, top10['Abs_Change'] / 1e12, color='darkred', edgecolor='black')
plt.xlabel("Absolute Change in Output Multiplier (Trillions)", fontsize=12)
plt.title("Top 10 Most Negatively Affected Sectors\n(25% Tariff on Imports from China's Electronics Sector)", fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("tariff_impact_absolute_named_fixed2.png")
plt.close()

# === Bar Chart: Top 10 Most Positively Affected Sectors ===
top10_positive = results.sort_values(by='Abs_Change', ascending=False).head(10)
labels_positive = top10_positive['SectorName'] + ' (' + top10_positive['Country'] + ')'

plt.figure(figsize=(13, 8))
bars = plt.barh(labels_positive, top10_positive['Abs_Change'] / 1e12,
                color='darkgreen', edgecolor='black')

plt.xlabel("Absolute Change in Output Multiplier (Trillions)", fontsize=12)
plt.title("Top 10 Most Positively Affected Sectors\n(25% Tariff on Imports from China's Electronics Sector)",
          fontsize=14, pad=20)
plt.xticks(fontsize=10)
plt.yticks(fontsize=9)
plt.gca().invert_yaxis()
plt.subplots_adjust(left=0.35, right=0.95, top=0.88, bottom=0.12)
plt.tight_layout()
plt.savefig("tariff_impact_positive_top10_fixed.png", bbox_inches='tight')
plt.close()

# === Save Input-Output Tables for Top 12 Affected Sectors ===
print("Saving IO tables...")

# Ensure CHN_C26 is included in top sectors if it's not already
top_sectors = results.sort_values(by='Abs_Change', key=np.abs, ascending=False).head(12)['Sector'].tolist()
if 'CHN_C26' in A.columns and 'CHN_C26' not in top_sectors:
    top_sectors.append('CHN_C26')

# Get the subset of data
A_raw_subset = A.loc[top_sectors, top_sectors]
A_tariff_raw_subset = A_tariff.loc[top_sectors, top_sectors]

# === Hawkins-Simon Check for Subset ===
from numpy.linalg import LinAlgError

def check_hawkins_simon(matrix, label=""):
    n = matrix.shape[0]
    I_minus_A = np.eye(n) - matrix
    try:
        for k in range(1, n + 1):
            minor = I_minus_A[:k, :k]
            det = np.linalg.det(minor)
            if det <= 0:
                print(f"Hawkins-Simon condition violated at minor {k} of {label} (det = {det:.4e})")
                return False
        print(f"Hawkins-Simon condition satisfied for {label}: all leading principal minors positive")
        return True
    except LinAlgError as e:
        print(f"Error checking Hawkins-Simon condition for {label}:", str(e))
        return False

print("Checking Hawkins-Simon condition for selected top sectors...")
check_hawkins_simon(A_raw_subset.values, label="baseline subset")
check_hawkins_simon(A_tariff_raw_subset.values, label="tariff-adjusted subset")


# Scale for better visualization
A_subset = A_raw_subset * 1e3
A_tariff_subset = A_tariff_raw_subset * 1e3

def label_sector(sector_code):
    short_code = sector_code[4:]
    return f"{sector_code[:3]} - {sector_map.get(short_code, short_code)}"

A_subset.index = [label_sector(i) for i in A_subset.index]
A_subset.columns = [label_sector(i) for i in A_subset.columns]
A_tariff_subset.index = A_subset.index
A_tariff_subset.columns = A_subset.columns

# Create change mask (only for cells that actually changed)
change_mask = (A_raw_subset.values != A_tariff_raw_subset.values)

# Function to create heatmap with highlighted changes
def create_heatmap(matrix, title, filename, changes=None):
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues", cbar=False,
                     annot_kws={"size": 8}, linewidths=0.5)
    
    # Highlight changed cells if provided
    if changes is not None:
        for i in range(changes.shape[0]):
            for j in range(changes.shape[1]):
                if changes[i, j]:
                    ax.add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor='red', lw=2))
    
    plt.title(title, fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {filename}")

# Create both heatmaps
create_heatmap(A_subset, "Baseline IO Table (Top Affected Sectors)", "io_baseline_top12.png")
create_heatmap(A_tariff_subset, "Tariff-Adjusted IO Table (Top Affected Sectors)\nRed borders show tariff-affected cells", 
               "io_tariff_top12.png", change_mask)

# Additional visualization: Show just the differences
diff_matrix = (A_tariff_subset - A_subset)
plt.figure(figsize=(12, 10))
sns.heatmap(diff_matrix, annot=True, fmt=".2f", cmap="RdBu", center=0,
            annot_kws={"size": 8}, linewidths=0.5)
plt.title("Difference Between Tariff and Baseline IO Tables", fontsize=14, pad=20)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig("io_difference_top12.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: io_difference_top12.png")
