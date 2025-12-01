import os
import sys
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
    """
    Count the number of images per subdirectory.
    Returns a dictionary {class_name: num_images}
    """
    counts = {}
    subfolders = get_subdirectories(root_path)

    for folder in subfolders:
        class_name = os.path.basename(folder)
        images = get_images_in_folder(folder)
        counts[class_name] = len(images)
    
    return counts

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

    counts_dict = count_images_per_class(root)

    # Print number of images per class
    for class_name, count in counts_dict.items():
        print(f"{class_name} : {count} images")

    # Display charts
    plot_charts(counts_dict)

if __name__ == "__main__":
    main()
