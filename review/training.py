import os
from PIL import Image

def getImagesAndLabels(dataset_path):
    image_paths = []
    labels = []
    
    for filename in os.listdir(dataset_path):
        # Skip .DS_Store and any non-image files
        if filename == '.DS_Store':
            continue
        
        imagePath = os.path.join(dataset_path, filename)
        
        # Ensure the file is an image (you can add further checks if needed)
        try:
            PIL_img = Image.open(imagePath).convert('L')  # Convert to grayscale
            image_paths.append(PIL_img)
            labels.append(0)  # You can modify this to assign proper labels
        except Exception as e:
            print(f"Error processing image {filename}: {e}")
    
    return image_paths, labels
