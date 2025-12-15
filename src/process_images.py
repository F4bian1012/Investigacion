import cv2
import os
import glob
from pathlib import Path

# Paths
RAW_DIR = "data/raw/imagenes"
PROCESSED_DIR = "data/processed/grayscale"

def process_images():
    """
    Converts all images in data/raw/imagenes to grayscale
    and saves them to data/processed/grayscale.
    """
    # ensure output directory exists
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Supported extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    files = []
    
    print(f"🔍 Searching for images in {RAW_DIR}...")
    for ext in extensions:
        files.extend(glob.glob(os.path.join(RAW_DIR, ext)))
        
    if not files:
        print("⚠️ No images found! Please add images to data/raw/imagenes/")
        return

    print(f"📸 Found {len(files)} images. Starting processing...")
    
    processed_count = 0
    
    for file_path in files:
        try:
            # Read image
            img = cv2.imread(file_path)
            if img is None:
                print(f"❌ Failed to load: {file_path}")
                continue
                
            # Convert to Grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Save
            filename = os.path.basename(file_path)
            output_path = os.path.join(PROCESSED_DIR, filename)
            # 
            cv2.imwrite(output_path, gray)
            processed_count += 1
            print(f"✅ Processed: {filename}")
            
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")

    print("\n========================================")
    print(f"🎉 Processing Complete!")
    print(f"   Total Input: {len(files)}")
    print(f"   Successfully Converted: {processed_count}")
    print(f"   Output Folder: {PROCESSED_DIR}")
    print("========================================")

if __name__ == "__main__":
    process_images()
