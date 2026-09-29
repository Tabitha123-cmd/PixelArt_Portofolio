import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

def process_pixel_art(pixelbyme, num_colors=8, output_path="portfolio_result.png"):
    # 1. Check if image file exists
    if not os.path.exists(pixelbyme):
        raise FileNotFoundError(f"Image not found at path: '{pixelbyme}'")
        
    img = cv2.imread(pixelbyme)
    if img is None:
        raise ValueError(f"Failed to load image from '{pixelbyme}'. Ensure it is a valid image file.")
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, _ = img_rgb.shape

    pixels = img_rgb.reshape(-1, 3)
    unique_colors_count = len(np.unique(pixels, axis=0))
    
    # 2. Adjust num_colors if image has fewer unique colors
    actual_num_colors = min(num_colors, unique_colors_count)

    # 3. K-Means Color Quantization
    kmeans = KMeans(n_clusters=actual_num_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_

    quantized_pixels = colors[labels]
    quantized_img = quantized_pixels.reshape(img_rgb.shape).astype(np.uint8)

    counts = np.bincount(labels)
    percentages = counts / len(labels)

    # 4. Generate Palette Bar (Fixing Rounding / Truncation Gap)
    bar_width = 300
    palette_bar = np.zeros((50, bar_width, 3), dtype=np.uint8)
    cum_percentages = np.cumsum(percentages)
    
    start_x = 0
    for i, (cum_p, color) in enumerate(zip(cum_percentages, colors)):
        end_x = bar_width if i == len(colors) - 1 else int(cum_p * bar_width)
        palette_bar[:, start_x:end_x] = color
        start_x = end_x

    # 5. Segment Sprite (Assuming most frequent color is background)
    bg_color_index = np.argmax(counts)
    mask = (labels != bg_color_index).reshape((height, width))

    segmented_img = img_rgb.copy()
    # Mask background with magenta [255, 0, 255] or transparent RGBA to easily distinguish from black outlines
    segmented_img[~mask] = [255, 0, 255] 

    # 6. Plotting
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(img_rgb)
    axes[0].set_title("1. Original Pixel Art")
    axes[0].axis('off')

    axes[1].imshow(palette_bar)
    axes[1].set_title(f"2. Extracted Palette ({actual_num_colors} Colors)")
    axes[1].axis('off')

    axes[2].imshow(quantized_img)
    axes[2].set_title("3. Quantized Color Art")
    axes[2].axis('off')

    axes[3].imshow(segmented_img)
    axes[3].set_title("4. Segmented Sprite (Bg Removed)")
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

# Example Usage
if __name__ == "__main__":
    process_pixel_art("pixelbyme.png", num_colors=8)
