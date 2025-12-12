import cv2
import os
import glob
import argparse

# Default Paths
INPUT_DIR = "data/processed/grayscale"
PROCESSED_BASE_DIR = "data/processed"

def reshape_images(width, height):
    """
    Resizes images from INPUT_DIR and saves them to data/processed/{width}x{height}.
    """
    output_folder_name = f"{width}x{height}"
    output_dir = os.path.join(PROCESSED_BASE_DIR, output_folder_name)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Supported extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    files = []
    
    print(f"🔍 Searching for images in {INPUT_DIR}...")
    for ext in extensions:
        files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
        
    if not files:
        print(f"⚠️ No images found in {INPUT_DIR}")
        return

    print(f"📸 Found {len(files)} images. Resizing to {width}x{height}...")
    
    processed_count = 0
    
    for file_path in files:
        try:
            img = cv2.imread(file_path)
            if img is None:
                print(f"❌ Failed to load: {file_path}")
                continue
                
            # Resize
            resized = cv2.resize(img, (width, height))
            
            # Save
            filename = os.path.basename(file_path)
            output_path = os.path.join(output_dir, filename)
            
            cv2.imwrite(output_path, resized)
            processed_count += 1
            # print(f"✅ Processed: {filename}") # Verbose
            
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")

    print("\n========================================")
    print(f"🎉 Reshape Complete!")
    print(f"   Target Size: {width}x{height}")
    print(f"   Output Folder: {output_dir}")
    print(f"   Processed: {processed_count}/{len(files)}")
    print("========================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reshape images for TinyML")
    parser.add_argument("--width", type=int, default=28, help="Target width")
    parser.add_argument("--height", type=int, default=28, help="Target height")
    
    args = parser.parse_args()
    
    reshape_images(args.width, args.height)
