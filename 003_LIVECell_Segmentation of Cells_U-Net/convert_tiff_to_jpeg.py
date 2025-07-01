from pathlib import Path
from PIL import Image

# Define the folder containing the TIFF images
folder = Path(r"C:\Users\shali\Documents\L&D\GitHub Projects\Machine Learning\003_LIVECell_Segmentation of Cells_U-Net\images\test set jpeg")

# Loop through each .tif or .tiff file in the folder
for tif_file in folder.glob("*.tif*"):
    # Define the corresponding JPEG path
    jpg_file = tif_file.with_suffix(".jpg")
    
    # Open the TIFF image, convert to RGB (JPEG doesn't support transparency), and save as JPEG
    with Image.open(tif_file) as img:
        img.convert("RGB").save(jpg_file, "JPEG", quality=95, optimize=True)
    
    # Delete the original TIFF file
    tif_file.unlink()

print("✅ Conversion complete: All TIFF files converted to JPEG and original TIFFs deleted.")
