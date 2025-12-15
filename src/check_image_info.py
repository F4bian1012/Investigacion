
import cv2
import os
import sys

def check_image_info(image_path):
    """
    Checks and prints the dimensions and file size of an image at the given path.
    """
    if not os.path.exists(image_path):
        print(f"❌ Error: File not found at {image_path}")
        return

    try:
        # Get file size
        file_size_bytes = os.path.getsize(image_path)
        file_size_kb = file_size_bytes / 1024
        file_size_mb = file_size_kb / 1024

        # Read image to get dimensions
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"❌ Error: Could not read image at {image_path}. Check if it's a valid image file.")
            return

        print(f"\nℹ️  Image Info for: {image_path}")
        print("-" * 40)
        
        # Dimensions
        height, width = img.shape[:2]
        channels = img.shape[2] if len(img.shape) > 2 else 1
        
        print(f"📏 Dimensions: {width} x {height} px")
        print(f"🎨 Channels:   {channels}")
        
        # File Size
        print(f"💾 File Size:  {file_size_bytes} bytes")
        print(f"              {file_size_kb:.2f} KB")
        print(f"              {file_size_mb:.2f} MB")
        print("-" * 40 + "\n")

    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/check_image_info.py <path_to_image>")
        # Example usage/fallback: try to find an image in data/raw if available to demo
        pass
    else:
        path = sys.argv[1]
        check_image_info(path)
