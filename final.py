import pandas as pd
import numpy as np

# Define the sectors we're interested in (electronics-related)
ELECTRONICS_SECTORS = [
    'C26',  # Computer, electronic and optical products
    'C261', # Electronic components
    'C262', # Computers
    'C263', # Communication equipment
    'C264'  # Consumer electronics
]

# Countries we're analyzing
COUNTRIES = ['CHN', 'USA']

def load_and_filter_data(filepath):
    """Load the massive CSV and filter to only electronics sectors for China and US"""
    # Read the CSV (we only need the first row to understand the column structure)
    df = pd.read_csv(filepath, nrows=1)
    
    # Get all column names
    cols = df.columns.tolist()
    
    # Find columns that match our criteria (country + electronics sector)
    selected_cols = ['V1']  # Keep the row identifier column
    
    for country in COUNTRIES:
        for sector in ELECTRONICS_SECTORS:
            # Find columns that match the country and sector pattern
            pattern = f"{country}_{sector}"
            matching_cols = [c for c in cols if pattern in c]
            selected_cols.extend(matching_cols)
    
    # Now read the full data but only for selected columns
    full_data = pd.read_csv(filepath, usecols=selected_cols)
    
    return full_data

def create_io_matrix(data):
    """Create an input-output matrix from the filtered data"""
    # The first column 'V1' contains the row identifiers which match the column names
    # We'll set this as the index and transpose to get sectors as both rows and columns
    data.set_index('V1', inplace=True)
    io_matrix = data.T
    
    # Filter to only include our selected countries and sectors
    io_matrix = io_matrix.loc[[idx for idx in io_matrix.index 
                             if any(f"{c}_" in idx for c in COUNTRIES) 
                             and any(s in idx for s in ELECTRONICS_SECTORS)]]
    
    # Fill NA values with 0
    io_matrix = io_matrix.fillna(0)
    
    return io_matrix

def simulate_tariff_impact(io_matrix, tariff_rate=0.3, shock_country='USA', affected_sector='C26'):
    """
    Simulate the impact of a tariff on electronics trade between China and US
    
    Parameters:
    - io_matrix: The input-output matrix
    - tariff_rate: The tariff rate to apply (e.g., 0.3 for 30%)
    - shock_country: The country imposing the tariff ('USA' or 'CHN')
    - affected_sector: The sector being tariffed (default 'C26' for all electronics)
    """
    # Identify the trade flows to modify
    if shock_country == 'USA':
        exporting_country = 'CHN'
    else:
        exporting_country = 'USA'
    
    # Find the relevant trade flows
    tariff_cols = [col for col in io_matrix.columns 
                  if f"{exporting_country}_{affected_sector}" in col]
    
    shock_rows = [row for row in io_matrix.index 
                 if f"{shock_country}_" in row]
    
    # Apply the tariff by reducing the input coefficients
    for col in tariff_cols:
        for row in shock_rows:
            original_value = io_matrix.loc[row, col]
            io_matrix.loc[row, col] = original_value * (1 - tariff_rate)
    
    # Calculate the Leontief inverse matrix for impact analysis
    A = io_matrix.values  # Technical coefficients matrix
    I = np.identity(A.shape[0])  # Identity matrix
    L = np.linalg.inv(I - A)  # Leontief inverse
    
    # Calculate total output impacts
    initial_output = np.ones(A.shape[0])  # Assume unit initial output for simplicity
    impacted_output = L @ initial_output
    
    # Create a DataFrame to display results
    results = pd.DataFrame({
        'Sector': io_matrix.index,
        'Initial_Output': initial_output,
        'Impacted_Output': impacted_output,
        'Percentage_Change': (impacted_output - initial_output) / initial_output * 100
    })
    
    return results

# Main analysis
if __name__ == "__main__":
    # Load and process the data
    data = load_and_filter_data('2020_SML.csv')
    io_matrix = create_io_matrix(data)
    
    print("Input-Output Matrix Dimensions:", io_matrix.shape)
    print("\nSample of Input-Output Matrix:")
    print(io_matrix.head())
    
    # Simulate a 30% US tariff on Chinese electronics
    print("\nSimulating 30% US tariff on Chinese electronics...")
    results = simulate_tariff_impact(io_matrix, tariff_rate=0.3)
    
    print("\nImpact Results:")
    print(results.sort_values('Percentage_Change', ascending=False))
