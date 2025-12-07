import rasterio
import numpy as np
import os
import glob

# --- CONFIGURATION ---
# Path where your individual GeoTIFF band files are located
INPUT_DIR = './deployment_data/input_bands/' 
OUTPUT_FILE = './deployment_data/final_9band_patch.tif'

# Define the 8 spectral bands + 1 NDVI needed for the model
# NOTE: This order must match your training data (B2, B3, B4, B8 for Before/After)
BAND_FILES = [
    'before_B2.tif', 'before_B3.tif', 'before_B4.tif', 'before_B8.tif',
    'after_B2.tif', 'after_B3.tif', 'after_B4.tif', 'after_B8.tif',
    'ndvi_band.tif' # This must be the 9th band
]

def stack_bands_to_single_tif(input_dir, output_file, band_names):
    """
    Loads individual TIFF bands and stacks them into a single GeoTIFF file.
    All bands must have the same resolution (10m) and dimensions (256x256).
    """
    file_paths = [os.path.join(input_dir, name) for name in band_names]
    
    # 1. Read the data from all individual bands
    source_bands = []
    
    # Get profile (metadata) from the first file
    with rasterio.open(file_paths[0]) as src:
        profile = src.profile
        # Read data from the first band
        source_bands.append(src.read(1))

    # Read remaining bands
    for filepath in file_paths[1:]:
        with rasterio.open(filepath) as src:
            source_bands.append(src.read(1))
    
    # 2. Stack the bands into a NumPy array
    stacked_array = np.stack(source_bands, axis=0)
    
    # 3. Update the metadata profile for the new stacked file
    profile.update(
        dtype=rasterio.float32,
        count=len(band_names),  # Set the band count to 9
        nodata=0 # Assuming 0 is the no-data value
    )
    
    # 4. Write the new multi-band TIFF file
    with rasterio.open(output_file, 'w', **profile) as dst:
        # Write the entire 9-band array
        dst.write(stacked_array)

    print(f"\nSuccessfully created 9-band GeoTIFF: {output_file}")
    print(f"Final shape: {stacked_array.shape}")
    
# --- Execution ---
if __name__ == "__main__":
    # You would typically generate the individual band TIFs and the NDVI TIF 
    # from your exported Earth Engine data first, then run this stacker.
    
    # Check if the required number of files exist before stacking
    files_found = glob.glob(os.path.join(INPUT_DIR, '*.tif'))
    if len(files_found) < len(BAND_FILES):
        print(f"Error: Only {len(files_found)} files found. Need {len(BAND_FILES)} files.")
    else:
        stack_bands_to_single_tif(INPUT_DIR, OUTPUT_FILE, BAND_FILES)