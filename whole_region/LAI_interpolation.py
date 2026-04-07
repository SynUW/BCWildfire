import os
import shutil
from datetime import datetime, timedelta
from tqdm import tqdm


def fill_missing_lai_data(source_folder, output_folder):
    """
    Fills missing LAI data by copying the nearest neighbor's data.
    Skips files that already exist in the output folder.

    Args:
        source_folder (str): Path to the folder containing the original LAI data files.
        output_folder (str): Path to the folder where the complete dataset will be stored.
    """

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # List all files in both source and output folders
    source_files = os.listdir(source_folder)
    output_files = set(os.listdir(output_folder)) if os.path.exists(output_folder) else set()

    # Filter for LAI files and extract dates
    lai_files = [f for f in source_files if f.startswith("clipped_LAI_") and f.endswith(".tif")]
    existing_dates = set()
    for filename in lai_files:
        try:
            date_str = filename[12:-4]
            date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
            existing_dates.add(date_obj)
        except ValueError:
            print(f"Warning: Could not parse date from filename: {filename}")

    if not existing_dates:
        print("No valid LAI files found in the source folder.")
        return

    # Find the start and end date of the existing data
    start_date = min(existing_dates)
    end_date = max(existing_dates)

    # Generate the full date range
    full_date_range = set()
    current_date = start_date
    while current_date <= end_date:
        full_date_range.add(current_date)
        current_date += timedelta(days=1)

    # Identify missing dates
    missing_dates = sorted(list(full_date_range - existing_dates))

    print(f"Found {len(lai_files)} original LAI files.")
    print(f"Time range: {start_date} to {end_date}")
    print(f"Number of missing dates: {len(missing_dates)}")

    # Copy existing files to the new folder (only if they don't exist)
    copied_count = 0
    skipped_count = 0
    for filename in lai_files:
        destination_path = os.path.join(output_folder, filename)
        if filename not in output_files:
            source_path = os.path.join(source_folder, filename)
            shutil.copy2(source_path, destination_path)  # Use copy2 to preserve metadata
            copied_count += 1
        else:
            skipped_count += 1
    print(f"Copied {copied_count} existing files, skipped {skipped_count} already existing files")

    # Fill in missing data
    filled_count = 0
    skipped_missing_count = 0
    for missing_date in tqdm(missing_dates, desc="Filling missing data"):
        missing_filename = f"clipped_LAI_{missing_date.strftime('%Y-%m-%d')}.tif"
        destination_path = os.path.join(output_folder, missing_filename)

        # Skip if file already exists
        if missing_filename in output_files:
            skipped_missing_count += 1
            continue

        # Find the nearest existing date
        nearest_date = None
        min_difference = None  # Initialize as None instead of infinity

        sorted_existing_dates = sorted(list(existing_dates))

        for existing_date in sorted_existing_dates:
            difference = abs((missing_date - existing_date).days)  # Calculate difference in days
            if min_difference is None or difference < min_difference:
                min_difference = difference
                nearest_date = existing_date
            elif difference == min_difference:
                # If the difference is the same, choose the closer one in time
                if abs((missing_date - existing_date).days) < abs((missing_date - nearest_date).days):
                    nearest_date = existing_date

        if nearest_date:
            # Construct source filename
            nearest_filename = f"clipped_LAI_{nearest_date.strftime('%Y-%m-%d')}.tif"
            source_path = os.path.join(source_folder, nearest_filename)

            try:
                shutil.copy2(source_path, destination_path)
                filled_count += 1
            except FileNotFoundError:
                print(f"Error: Nearest neighbor file not found: {source_path}")
        else:
            print(f"Warning: Could not find nearest neighbor for {missing_date}")

    print(f"\nFinished processing:")
    print(f"- Filled {filled_count} missing dates")
    print(f"- Skipped {skipped_missing_count} already existing missing date files")
    print(f"Complete dataset stored in: {output_folder}")


if __name__ == "__main__":
    input_folder = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/LAI_LULC/LAI_BCBoundingbox'
    output_folder = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/LAI_LULC/LAI_BCBoundingbox_interpolated'

    fill_missing_lai_data(input_folder, output_folder)