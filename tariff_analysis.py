import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# -----------------------------------------------------------------------------
# --- Helper Functions for Plotting (Features Restored) ---
# -----------------------------------------------------------------------------
def create_bar_chart(results_df, num_sectors, ascending, color, title_text, filename):
    plt.figure(figsize=(12, 8))
    top_data = results_df.sort_values(by='Abs_Change', ascending=ascending).head(num_sectors)
    labels = top_data['SectorName'] + ' (' + top_data['Country'] + ')'
    plt.barh(labels, top_data['Abs_Change'], color=color, edgecolor='black')
    plt.xlabel("Absolute Change in Output", fontsize=12)
    plt.title(title_text, fontsize=14, pad=20)
    plt.xticks(fontsize=10); plt.yticks(fontsize=10)
    plt.gca().invert_yaxis(); plt.tight_layout(pad=1)
    plt.savefig(filename, dpi=300, bbox_inches='tight'); plt.close()
    print(f"✅ Saved: {filename}")

def create_heatmap(matrix, title, filename, changes=None, cmap="Blues", center=None):
    plt.figure(figsize=(12, 10))
    # FEATURE RESTORED: annot=True and formatting for numbers
    ax = sns.heatmap(matrix, annot=True, fmt=".2f", cmap=cmap, center=center,
                     annot_kws={"size": 8}, linewidths=0.5)
    # FEATURE RESTORED: Red boxes for changed cells
    if changes is not None:
        rows, cols = np.where(changes)
        for r, c in zip(rows, cols):
             ax.add_patch(Rectangle((c, r), 1, 1, fill=False, edgecolor='red', lw=2))
    plt.title(title, fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight'); plt.close()
    print(f"✅ Saved: {filename}")

# --------------------------------------------------------------------------
# --- Main Analysis Script ---
# --------------------------------------------------------------------------

# === Step 1: CORRECTLY Load and Parse the IO Data ===
print("Loading and correctly parsing IO data...")
try:
    full_data = pd.read_csv("2020_SML.csv", index_col=0)
    xls = pd.ExcelFile("ReadMe_ICIO_small.xlsx")
except FileNotFoundError as e:
    print(f"Error: {e}"); exit()

sector_cols = [col for col in full_data.columns if '_C' in col]
final_demand_cols = [col for col in full_data.columns if '_C' not in col]
country_sectors = [s for s in sector_cols if s.startswith('CHN_') or s.startswith('USA_')]
country_final_demand = [s for s in final_demand_cols if s.startswith('CHN_') or s.startswith('USA_')]
Z = full_data.loc[country_sectors, country_sectors]
Y = full_data.loc[country_sectors, country_final_demand]
X = Z.sum(axis=1) + Y.sum(axis=1)
X[X == 0] = 1e-9

# === Step 2: CORRECTLY Build the Leontief Model ===
print("Building corrected Leontief model...")
A = Z.div(X, axis=1)
d_base = Y.sum(axis=1)
I = np.eye(len(A))
L_base = np.linalg.inv(I - A.values)
output_base = L_base @ d_base

# === Step 3: Apply Tariff Directly to Coefficients (Conceptually Flawed Method) ===
print("Applying direct tariff to coefficients...")
# WARNING: This method is economically unstable.
A_tariff = A.copy()
tariff_row = 'CHN_C26'
usa_cols = [col for col in A.columns if col.startswith('USA_')]
A_tariff.loc[tariff_row, usa_cols] *= 1.25

# === Step 4: Leontief Inverse and Output Calculation ===
print("Calculating Leontief inverse and output changes...")
try:
    L_tariff = np.linalg.inv(I - A_tariff.values)
except np.linalg.LinAlgError:
    print("FATAL ERROR: Tariff-adjusted matrix is singular. The model is not economically viable.")
    L_tariff = L_base

output_tariff = L_tariff @ d_base
abs_change = output_tariff - output_base
output_base[output_base == 0] = 1e-9
percent_change = (abs_change / output_base) * 100

# === Step 5: Create and Save Results DataFrame ===
print("Creating results dataframe...")
results = pd.DataFrame({
    'Sector': A.columns, 'Baseline': output_base, 'PostTariff': output_tariff,
    'Abs_Change': abs_change, 'Pct_Change': percent_change
})
results['Country'] = results['Sector'].str[:3]
results['SectorCode'] = results['Sector'].str[4:]
sector_info = xls.parse('Area_Activities').iloc[2:, [7, 8, 9]]
sector_info.columns = ['Old_Code', 'Code', 'Sector_Name']
sector_info = sector_info.dropna(subset=['Code', 'Sector_Name'])
sector_map = dict(zip(sector_info['Code'].str.extract(r'([A-Z0-9_]+)$')[0], sector_info['Sector_Name']))
results['SectorName'] = results['SectorCode'].map(sector_map).fillna(results['SectorCode'])
results.to_csv("tariff_results_direct_model.csv", index=False)
print("✅ Saved: tariff_results_direct_model.csv")

# === Step 6: Generate Visualizations ===
print("Generating visualizations...")

# FEATURE RESTORED: Bar chart for Top 10 Positive Movers
create_bar_chart(results, 10, False, 'darkgreen', "Top 10 Positively Affected Sectors (Direct Tariff Model)", "impact_positive_direct_model.png")
create_bar_chart(results, 10, True, 'darkred', "Top 10 Most Negatively Affected Sectors (Direct Tariff Model)", "impact_negative_direct_model.png")

# Heatmap Visualization
top_sectors = results.sort_values(by='Abs_Change', key=np.abs, ascending=False).head(12)['Sector'].tolist()
if 'CHN_C26' in A.columns and 'CHN_C26' not in top_sectors:
    top_sectors.append('CHN_C26')

A_subset = A.loc[top_sectors, top_sectors]
A_tariff_subset = A_tariff.loc[top_sectors, top_sectors]
change_mask = (A.loc[top_sectors, top_sectors].values != A_tariff.loc[top_sectors, top_sectors].values)

# Create heatmaps using the restored function
create_heatmap(A_subset, "Baseline Coefficients (Top Affected Sectors)", "heatmap_baseline_direct_model.png")
create_heatmap(A_tariff_subset, "Tariff-Adjusted Coefficients (Top Affected Sectors)", "heatmap_tariff_direct_model.png", changes=change_mask)
create_heatmap(A_tariff_subset - A_subset, "Difference in Coefficients", "heatmap_difference_direct_model.png", cmap="RdBu", center=0)

print("\n--- Analysis Complete ---")