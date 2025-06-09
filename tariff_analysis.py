import matplotlib
matplotlib.use('Agg')  # Prevent GUI issues in headless environments

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# -----------------------------------------------------------------------------
# --- Helper Functions for Plotting (Refactored to Reduce Repetition) ---
# -----------------------------------------------------------------------------

def create_bar_chart(results_df, num_sectors, ascending, color, title_text, filename):
    """Creates and saves a horizontal bar chart for top N affected sectors."""
    plt.figure(figsize=(12, 8))
    
    # Sort data and select top sectors
    top_data = results_df.sort_values(by='Abs_Change', ascending=ascending).head(num_sectors)
    labels = top_data['SectorName'] + ' (' + top_data['Country'] + ')'
    
    # Create plot
    plt.barh(labels, top_data['Abs_Change'], color=color, edgecolor='black')
    plt.xlabel("Absolute Change in Output Multiplier", fontsize=12)
    plt.title(title_text, fontsize=14, pad=20)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.gca().invert_yaxis()
    plt.tight_layout(pad=1)
    
    # Save and close
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {filename}")

def create_heatmap(matrix, title, filename, changes=None, cmap="Blues", center=None):
    """Creates and saves a heatmap."""
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(matrix, annot=True, fmt=".2f", cmap=cmap, center=center, cbar=False,
                     annot_kws={"size": 8}, linewidths=0.5)

    if changes is not None:
        rows, cols = np.where(changes)
        for r, c in zip(rows, cols):
             ax.add_patch(Rectangle((c, r), 1, 1, fill=False, edgecolor='red', lw=2))

    plt.title(title, fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {filename}")

def check_hawkins_simon(matrix, label=""):
    """Checks the Hawkins-Simon condition for economic viability."""
    n = matrix.shape[0]
    I_minus_A = np.eye(n) - matrix
    try:
        for k in range(1, n + 1):
            if np.linalg.det(I_minus_A[:k, :k]) <= 0:
                print(f"Hawkins-Simon condition VIOLATED at minor {k} of {label}")
                return False
        print(f"Hawkins-Simon condition SATISFIED for {label}")
        return True
    except np.linalg.LinAlgError as e:
        print(f"Error checking Hawkins-Simon condition for {label}: {e}")
        return False

# --------------------------------------------------------------------------
# --- Main Analysis Script ---
# --------------------------------------------------------------------------

# === Step 1: Load and Filter Data ===
print("Loading CSV data...")
data = pd.read_csv("2020_SML.csv", index_col=0)
chn_usa_sectors = [s for s in data.index if s.startswith('CHN_') or s.startswith('USA_')]
data = data.loc[chn_usa_sectors, chn_usa_sectors]
common_sectors = data.index.intersection(data.columns)
data = data.loc[common_sectors, common_sectors]

# === Step 2: Normalize to Technical Coefficient Matrix A ===
col_sums = data.sum(axis=0)
col_sums[col_sums.abs() < 1e-9] = 1e-9 # Use a small number to prevent division by zero
A = data.divide(col_sums, axis=1)

# === Step 3: Apply 25% Tariff (Corrected Logic) ===
print("Applying tariff...")
A_tariff = A.copy()
tariff_row = 'CHN_C26'
usa_cols = [col for col in A.columns if col.startswith('USA_')]
if tariff_row in A.index and usa_cols:
    A_tariff.loc[tariff_row, usa_cols] *= 1.25

# === Step 4: Leontief Inverse and Output Calculation ===
print("Calculating Leontief inverse and output changes...")
I = np.eye(len(A))
L_base = np.linalg.inv(I - A.values)

try:
    L_tariff = np.linalg.inv(I - A_tariff.values)
except np.linalg.LinAlgError:
    print("Error: Tariff-adjusted matrix is singular. The Hawkins-Simon condition is likely violated.")
    L_tariff = L_base

final_demand = np.ones((len(A), 1)) * 1e-6
output_base = L_base @ final_demand
output_tariff = L_tariff @ final_demand
abs_change = output_tariff - output_base
percent_change = (abs_change / output_base) * 100

# === Step 5: Create and Save Results DataFrame ===
print("Creating results dataframe...")
results = pd.DataFrame({
    'Sector': A.columns, 'Baseline': output_base.flatten(), 'PostTariff': output_tariff.flatten(),
    'Abs_Change': abs_change.flatten(), 'Pct_Change': percent_change.flatten()
})
results['Country'] = results['Sector'].str[:3]
results['SectorCode'] = results['Sector'].str[4:]

# Load sector names from Excel
xls = pd.ExcelFile("ReadMe_ICIO_small.xlsx")
df = xls.parse('Area_Activities')
sector_info = df.iloc[2:, [7, 8, 9]]
sector_info.columns = ['Old_Code', 'Code', 'Sector_Name']
sector_info = sector_info.dropna(subset=['Code', 'Sector_Name'])
sector_map = dict(zip(sector_info['Code'].str.extract(r'([A-Z0-9_]+)$')[0], sector_info['Sector_Name']))
results['SectorName'] = results['SectorCode'].map(sector_map).fillna(results['SectorCode'])

results.to_csv("tariff_results.csv", index=False)
print("✅ Saved: tariff_results.csv")

# === Step 6: Generate Visualizations ===
print("Generating visualizations...")

# Create bar charts using the refactored function
create_bar_chart(
    results_df=results, num_sectors=10, ascending=True, color='darkred',
    title_text="Top 10 Most Negatively Affected Sectors\n(25% Tariff on US Imports from China's Electronics Sector)",
    filename="tariff_impact_negative.png"
)
create_bar_chart(
    results_df=results, num_sectors=10, ascending=False, color='darkgreen',
    title_text="Top 10 Most Positively Affected Sectors\n(25% Tariff on US Imports from China's Electronics Sector)",
    filename="tariff_impact_positive.png"
)

# Prepare data for heatmaps
top_sectors = results.sort_values(by='Abs_Change', key=np.abs, ascending=False).head(12)['Sector'].tolist()
if 'CHN_C26' in A.columns and 'CHN_C26' not in top_sectors:
    top_sectors.append('CHN_C26')

A_raw_subset = A.loc[top_sectors, top_sectors]
A_tariff_raw_subset = A_tariff.loc[top_sectors, top_sectors]

# Perform Hawkins-Simon check on the subset
print("Checking Hawkins-Simon condition for selected top sectors...")
check_hawkins_simon(A_raw_subset.values, label="baseline subset")
check_hawkins_simon(A_tariff_raw_subset.values, label="tariff-adjusted subset")

# Create heatmaps
def label_sector(sector_code):
    return f"{sector_code[:3]} - {sector_map.get(sector_code[4:], sector_code[4:])}"

A_subset = (A_raw_subset * 1e3)
A_subset.index = [label_sector(i) for i in A_subset.index]
A_subset.columns = [label_sector(i) for i in A_subset.columns]

A_tariff_subset = (A_tariff_raw_subset * 1e3)
A_tariff_subset.index = A_subset.index
A_tariff_subset.columns = A_subset.columns

change_mask = (A_raw_subset.values != A_tariff_raw_subset.values)
diff_matrix = A_tariff_subset - A_subset

create_heatmap(A_subset, "Baseline IO Table (Top Affected Sectors)", "io_baseline_top12.png")
create_heatmap(A_tariff_subset, "Tariff-Adjusted IO Table (Top Affected Sectors)\nRed borders show tariff-affected cells", "io_tariff_top12.png", changes=change_mask)
create_heatmap(diff_matrix, "Difference Between Tariff and Baseline IO Tables", "io_difference_top12.png", cmap="RdBu", center=0)

print("\n--- Analysis Complete ---")