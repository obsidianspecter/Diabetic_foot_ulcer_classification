import os
import shutil
import pandas as pd
import re

def organize_dataset():
    # Source and destination directories
    src_dir = "local_database_Processed (2)"
    dest_dir = "local_database_Processed"
    
    # Create destination directories if they don't exist
    for severity in ['normal', 'moderate', 'severe']:
        for img_type in ['rgb', 'thermal']:
            os.makedirs(os.path.join(dest_dir, severity, img_type), exist_ok=True)
    
    # Process each patient directory
    for patient_dir in os.listdir(src_dir):
        if not os.path.isdir(os.path.join(src_dir, patient_dir)) or patient_dir == "__pycache__":
            continue
            
        patient_path = os.path.join(src_dir, patient_dir)
        
        # Process RGB and thermal images for each time point
        for file in os.listdir(patient_path):
            if not os.path.isfile(os.path.join(patient_path, file)):
                continue
                
            # Skip mask and depth images
            if "Mask" in file or "Depth" in file:
                continue
            
            # Process RGB images
            if "_RGB_T" in file and file.endswith(".png"):
                # Extract time point using regex
                match = re.search(r'T(\d+)', file)
                if match:
                    time_point = int(match.group(1))
                    
                    # Classify severity based on time point
                    if time_point <= 5:
                        severity = "normal"
                    elif time_point <= 10:
                        severity = "moderate"
                    else:
                        severity = "severe"
                    
                    # Copy RGB image
                    src_path = os.path.join(patient_path, file)
                    dest_path = os.path.join(dest_dir, severity, "rgb", f"{patient_dir}_{time_point}.png")
                    shutil.copy2(src_path, dest_path)
                    
                    # Find and copy corresponding thermal image
                    thermal_file = file.replace("RGB", "Tmap").replace(".png", ".tiff")
                    if os.path.exists(os.path.join(patient_path, thermal_file)):
                        src_thermal = os.path.join(patient_path, thermal_file)
                        dest_thermal = os.path.join(dest_dir, severity, "thermal", f"{patient_dir}_{time_point}.tiff")
                        shutil.copy2(src_thermal, dest_thermal)

if __name__ == "__main__":
    organize_dataset()
    print("Dataset organization completed!") 