import pandas as pd
import numpy as np
import matplotlib

# Set matplotlib backend to avoid Qt-related errors
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.stats import norm
import warnings
import os

warnings.filterwarnings('ignore')

# Set font to Arial
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False


def plot_selected_gaussian_distributions(excel_file_path, output_filename='selected_gaussian_distributions.png'):
    """
    Reads an Excel file and plots selected Gaussian distributions in a single row.
    Only plots specific driver factors: Band 20 Day, Band 20 Night, LAI, Total Precipitation,
    Volumetric Soil Water L1, and Volumetric Soil Water L4.

    Parameters:
    excel_file_path: Path to the Excel file.
    output_filename: Output image filename.

    Excel file format requirements:
    - Column 1: Driver factor name
    - Column 2: Burned mean
    - Column 3: Burned standard deviation
    - Column 4: Unburned mean
    - Column 5: Unburned standard deviation
    - Row 1: Header row (will be skipped)
    """

    try:
        # Read Excel file
        df = pd.read_excel(excel_file_path, header=0)

        # Ensure correct number of columns
        if df.shape[1] < 5:
            raise ValueError("The Excel file must have at least 5 columns of data.")

        # Rename columns for easier processing
        df.columns = ['驱动因素', 'Burned均值', 'Burned标准差', 'Unburned均值', 'Unburned标准差']

        # Delete empty rows
        df = df.dropna()

        # Define the selected driver factors to plot
        selected_factors = [
            'band 20 day', 'band 20 night', 'lai', 'total precipitation',
            'volumetric soil water l1', 'volumetric soil water l4'
        ]

        # Filter dataframe for selected factors (case-insensitive matching)
        filtered_df = df[df['驱动因素'].str.lower().isin(selected_factors)].copy()

        if len(filtered_df) == 0:
            print("No matching driver factors found.")
            return

        # Sort the dataframe to match the desired order
        factor_order = {factor: i for i, factor in enumerate(selected_factors)}
        filtered_df['order'] = filtered_df['驱动因素'].str.lower().map(factor_order)
        filtered_df = filtered_df.sort_values('order').drop('order', axis=1)

        n_factors = len(filtered_df)
        print(f"Found {n_factors} matching factors to plot.")

        # Create figure with subplots in a single row
        # Adjusted for A4 paper size with square-like subplots
        fig, axes = plt.subplots(1, n_factors, figsize=(n_factors * 1.8, 2.0))

        # If there's only one subplot, ensure axes is an array
        if n_factors == 1:
            axes = [axes]

        # Plot for each selected driver factor
        for i, (idx, row) in enumerate(filtered_df.iterrows()):
            ax = axes[i]

            factor_name = row['驱动因素']
            mean1, std1 = row['Burned均值'], row['Burned标准差']
            mean2, std2 = row['Unburned均值'], row['Unburned标准差']

            # Determine x-axis range
            x_min = min(mean1 - 4 * std1, mean2 - 4 * std2)
            x_max = max(mean1 + 4 * std1, mean2 + 4 * std2)

            # Generate x values
            x = np.linspace(x_min, x_max, 1000)

            # Calculate Probability Density
            y1 = norm.pdf(x, mean1, std1)
            y2 = norm.pdf(x, mean2, std2)

            # Plot curves
            ax.plot(x, y1, 'b-', linewidth=1.5, label='Burned')
            ax.plot(x, y2, 'r-', linewidth=1.5, label='Not Burned')

            # Fill area under curves
            ax.fill_between(x, y1, alpha=0.3, color='blue')
            ax.fill_between(x, y2, alpha=0.3, color='red')

            # Remove axis labels and y-axis ticks for cleaner look in single row
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_yticks([])

            # Only add legend for the last subplot (avoid overlap with curves)
            if i == 0:  # Last subplot
                legend = ax.legend(fontsize=6, loc='upper right', bbox_to_anchor=(1.035, 1))
                legend.get_frame().set_facecolor('none')  # Remove background
                legend.get_frame().set_edgecolor('none')  # Remove border

            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', which='major', labelsize=7)

            # Add title below the subplot
            ax.set_title(factor_name, fontsize=9, fontweight='bold', pad=-20)

        # Adjust layout to fit A4 paper
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Make room for titles below

        # Save image
        plt.savefig(output_filename, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()

        print(f"Combined plot saved as: {output_filename}")

        # Print data summary
        print("\nSelected Factors Data Summary:")
        print("-" * 60)
        for idx, row in filtered_df.iterrows():
            print(f"Driver Factor: {row['驱动因素']}")
            print(f"  Burned: Mean={row['Burned均值']:.3f}, Std Dev={row['Burned标准差']:.3f}")
            print(f"  Unburned: Mean={row['Unburned均值']:.3f}, Std Dev={row['Unburned标准差']:.3f}")
            print()

    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None


def plot_individual_gaussian_distributions(excel_file_path, output_dir='gaussian_plots'):
    """
    Original function kept for backward compatibility.
    Now calls the new selected plotting function.
    """
    return plot_selected_gaussian_distributions(excel_file_path)


def plot_gaussian_distributions(excel_file_path):
    """
    保持原有的函数名，供向后兼容使用
    """
    return plot_selected_gaussian_distributions(excel_file_path)


def save_plots_to_file(excel_file_path, output_filename='selected_gaussian_plots.png', dpi=300):
    """
    保存选定的高斯分布图到单个文件

    Parameters:
    excel_file_path: Path to the Excel file.
    output_filename: Output image filename.
    dpi: Image resolution.
    """
    return plot_selected_gaussian_distributions(excel_file_path, output_filename)


# Example usage
if __name__ == "__main__":
    # If you have your own Excel file, use the code below
    excel_file = r"D:\OneDrive - University of Calgary\Desktop\pixel_driver_distribution_results\distribution-只有均值和标准差-without-topo.xlsx"  # Replace with your Excel file path

    # Generate combined plot for selected driver factors
    plot_selected_gaussian_distributions(excel_file, output_filename='selected_factors_a4.png')

    # Alternative usage
    # save_plots_to_file(excel_file, output_filename='my_selected_plots.png')