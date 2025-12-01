import cv2
import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

def gaussian_blur(img):
    """Applique un flou gaussien"""
    return cv2.GaussianBlur(img, (15, 15), 0)

def create_mask(img):
    """Crée un masque binaire de la feuille"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Masque pour détecter le vert (feuille)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    # Nettoyage du masque
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def roi_objects(img, mask):
    """Détecte les objets d'intérêt (ROI)"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = img.copy()
    if contours:
        # Trouve le plus grand contour (la feuille)
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(result, [largest_contour], -1, (0, 255, 0), 3)
    return result

def analyze_object(img, mask):
    """Analyse l'objet: centre de masse, axes principaux"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = img.copy()

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        # Moments pour centre de masse
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(result, (cx, cy), 10, (0, 0, 255), -1)

        # Ellipse d'ajustement
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            cv2.ellipse(result, ellipse, (255, 0, 0), 2)

        # Rectangle englobant
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 255), 2)

    return result

def pseudolandmarks(img, mask):
    """Détecte les pseudo-landmarks sur le contour"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = img.copy()

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        # Simplification du contour
        epsilon = 0.01 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        # Échantillonnage uniforme de points sur le contour
        num_landmarks = 20
        step = max(1, len(largest_contour) // num_landmarks)
        landmarks = largest_contour[::step]

        # Dessine les landmarks
        for point in landmarks:
            cv2.circle(result, tuple(point[0]), 5, (255, 0, 255), -1)

    return result

def color_histogram(img):
    """Crée un histogramme de couleurs"""
    # Calcul des histogrammes pour chaque canal
    colors = ('b', 'g', 'r')
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, color in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        ax.plot(hist, color=color, label=color.upper())

    ax.set_xlim([0, 256])
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Color Histogram')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Convertit la figure en image
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    histogram_img = np.asarray(buf)
    histogram_img = cv2.cvtColor(histogram_img, cv2.COLOR_RGBA2BGR)
    plt.close(fig)

    return histogram_img

def process_image(img_path, save_dir=None):
    """Traite une image et retourne/sauvegarde les transformations"""
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Cannot read image {img_path}")
        return None

    base_name = os.path.splitext(os.path.basename(img_path))[0]

    # Génère toutes les transformations
    transformations = {
        'original': img,
        'gaussian_blur': gaussian_blur(img),
        'mask': None,  # sera traité après
        'roi_objects': None,
        'analyze_object': None,
        'pseudolandmarks': None,
        'color_histogram': color_histogram(img)
    }

    # Crée le masque (nécessaire pour les autres transformations)
    mask = create_mask(img)
    transformations['mask'] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    transformations['roi_objects'] = roi_objects(img, mask)
    transformations['analyze_object'] = analyze_object(img, mask)
    transformations['pseudolandmarks'] = pseudolandmarks(img, mask)

    if save_dir:
        # Mode: sauvegarde dans un dossier
        os.makedirs(save_dir, exist_ok=True)
        for name, trans_img in transformations.items():
            if trans_img is not None:
                output_path = os.path.join(save_dir, f"{base_name}_{name}.jpg")
                cv2.imwrite(output_path, trans_img)
        print(f"Transformations saved to {save_dir}")
    else:
        # Mode: affichage
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        titles = ['Original', 'Gaussian Blur', 'Mask', 'ROI Objects',
                  'Analyze Object', 'Pseudolandmarks', 'Color Histogram']

        for idx, (name, trans_img) in enumerate(transformations.items()):
            if trans_img is not None and idx < len(axes):
                # Convertit BGR en RGB pour matplotlib
                if name == 'color_histogram':
                    display_img = cv2.cvtColor(trans_img, cv2.COLOR_BGR2RGB)
                elif name == 'mask':
                    display_img = trans_img
                else:
                    display_img = cv2.cvtColor(trans_img, cv2.COLOR_BGR2RGB)

                axes[idx].imshow(display_img)
                axes[idx].set_title(titles[idx])
                axes[idx].axis('off')

        # Cache le dernier subplot vide
        axes[-1].axis('off')

        plt.tight_layout()
        plt.show()

    return transformations

def process_directory(src_dir, dst_dir):
    """Traite tous les images d'un dossier"""
    if not os.path.isdir(src_dir):
        print(f"Error: {src_dir} is not a directory")
        return

    # Trouve toutes les images
    valid_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_files = []

    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if any(file.endswith(ext) for ext in valid_extensions):
                image_files.append(os.path.join(root, file))

    if not image_files:
        print(f"No images found in {src_dir}")
        return

    print(f"Found {len(image_files)} images. Processing...")

    for img_path in image_files:
        print(f"Processing {img_path}...")
        # Préserve la structure de dossiers
        rel_path = os.path.relpath(img_path, src_dir)
        rel_dir = os.path.dirname(rel_path)
        save_dir = os.path.join(dst_dir, rel_dir) if rel_dir else dst_dir

        process_image(img_path, save_dir)

    print(f"All transformations saved to {dst_dir}")

def main():
    parser = argparse.ArgumentParser(description='Image transformation tool for leaf analysis')
    parser.add_argument('image_path', nargs='?', help='Path to a single image')
    parser.add_argument('-src', '--source', help='Source directory containing images')
    parser.add_argument('-dst', '--destination', help='Destination directory for transformed images')

    args = parser.parse_args()

    if args.source and args.destination:
        # Mode: traitement de dossier
        process_directory(args.source, args.destination)
    elif args.image_path:
        # Mode: traitement d'une seule image (affichage)
        if os.path.isfile(args.image_path):
            process_image(args.image_path)
        else:
            print(f"Error: {args.image_path} is not a valid file")
    else:
        print("Usage:")
        print("  Display transformations: python Transformation.py <image_path>")
        print("  Save transformations: python Transformation.py -src <source_dir> -dst <destination_dir>")
        parser.print_help()

if __name__ == "__main__":
    main()
