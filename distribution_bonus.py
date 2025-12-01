import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}

def get_subdirectories(root_path):
    """Return the list of subdirectories in root_path"""
    return [
        os.path.join(root_path, name)
        for name in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, name))
    ]

def get_images_in_folder(folder_path):
    """Return the list of images in a folder"""
    images = []
    for file_name in os.listdir(folder_path):
        ext = os.path.splitext(file_name)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            images.append(os.path.join(folder_path, file_name))
    return images

def count_images_per_class(root_path):
    """Count the number of images per subdirectory"""
    counts = {}
    subfolders = get_subdirectories(root_path)

    for folder in subfolders:
        class_name = os.path.basename(folder)
        images = get_images_in_folder(folder)
        counts[class_name] = len(images)
    
    return counts

def analyze_images_per_class(root_path):
    """
    Analyze images per class:
    - average width/height
    - average aspect ratio
    - mean brightness
    - mean saturation
    """
    analysis_results = {}
    subfolders = get_subdirectories(root_path)

    for folder in subfolders:
        class_name = os.path.basename(folder)
        images = get_images_in_folder(folder)

        if not images:
            continue

        widths, heights, ratios = [], [], []
        mean_brightness, mean_saturation = [], []
        hist_r, hist_g, hist_b = [], [], []

        for img_path in images:
            img = cv2.imread(img_path)
            if img is None:
                continue

            h, w = img.shape[:2]
            widths.append(w)
            heights.append(h)
            ratios.append(w / h)

            # Convert to HSV for brightness and saturation
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            brightness = np.mean(hsv[:, :, 2])
            saturation = np.mean(hsv[:, :, 1])
            mean_brightness.append(brightness)
            mean_saturation.append(saturation)

            # Histogramme RGB
            b, g, r = cv2.split(img)
            hist_b.append(np.mean(b))
            hist_g.append(np.mean(g))
            hist_r.append(np.mean(r))

        # Stocker les résultats moyens pour la classe
        analysis_results[class_name] = {
            "avg_width": np.mean(widths),
            "avg_height": np.mean(heights),
            "avg_ratio": np.mean(ratios),
            "mean_brightness": np.mean(mean_brightness),
            "mean_saturation": np.mean(mean_saturation),
            "mean_r": np.mean(hist_r),
            "mean_g": np.mean(hist_g),
            "mean_b": np.mean(hist_b)
        }

    return analysis_results

def plot_charts(counts_dict):
    """Display a bar chart and a pie chart from a dictionary {class_name: num_images}"""
    class_names = list(counts_dict.keys())
    counts = list(counts_dict.values())

    # --- Bar chart ---
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, counts)
    plt.title("Number of images per class")
    plt.xlabel("Class")
    plt.ylabel("Number of images")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # --- Pie chart ---
    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=class_names, autopct="%1.1f%%")
    plt.title("Dataset distribution")
    plt.show()

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 Distribution.py <directory>")
        sys.exit(1)
    
    root = sys.argv[1]

    if not os.path.isdir(root):
        print(f"Error: '{root}' is not a valid directory.")
        sys.exit(1)

    # Count images
    counts_dict = count_images_per_class(root)
    for class_name, count in counts_dict.items():
        print(f"{class_name} : {count} images")

    # Plot basic charts
    plot_charts(counts_dict)

    # Analyze images per class
    analysis_results = analyze_images_per_class(root)
    print("\n--- Image Analysis per Class ---")
    for class_name, stats in analysis_results.items():
        print(f"\nClass: {class_name}")
        print(f"Avg width: {stats['avg_width']:.2f}")
        print(f"Avg height: {stats['avg_height']:.2f}")
        print(f"Avg aspect ratio (W/H): {stats['avg_ratio']:.2f}")
        print(f"Mean brightness: {stats['mean_brightness']:.2f}")
        print(f"Mean saturation: {stats['mean_saturation']:.2f}")
        print(f"Mean R/G/B: ({stats['mean_r']:.1f}, {stats['mean_g']:.1f}, {stats['mean_b']:.1f})")

if __name__ == "__main__":
    main()
