import os
import sys
import argparse
import numpy as np
from plantcv import plantcv as pcv
import cv2

def gaussian_blur(img):
    """Applique un flou gaussien avec PlantCV"""
    return pcv.gaussian_blur(img=img, ksize=(15, 15))

def create_mask(img):
    """Crée un masque binaire de la feuille avec PlantCV"""
    # Conversion en LAB pour une meilleure segmentation
    a = pcv.rgb2gray_lab(rgb_img=img, channel='a')

    # Seuillage adaptatif
    mask = pcv.threshold.binary(gray_img=a, threshold=127, object_type='light')

    # Nettoyage avec morphologie
    mask = pcv.fill(bin_img=mask, size=100)
    mask = pcv.erode(gray_img=mask, ksize=3, i=1)
    mask = pcv.dilate(gray_img=mask, ksize=3, i=1)

    return mask

def roi_objects(img, mask):
    """Détecte les objets d'intérêt (ROI) avec PlantCV"""
    # Trouve les contours avec OpenCV
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dessine les contours
    result = img.copy()
    if len(contours) > 0:
        # Dessine tous les contours détectés
        cv2.drawContours(result, contours, -1, (0, 255, 0), 3)

    return result

def analyze_object(img, mask):
    """Analyse l'objet avec PlantCV: forme, taille, centre"""
    # Trouve les contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = img.copy()

    if len(contours) > 0:
        # Prend le plus grand contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Analyse avec PlantCV
        analysis = pcv.analyze.size(img=img, labeled_mask=mask)

        # Dessine le contour
        cv2.drawContours(result, [largest_contour], -1, (0, 255, 0), 3)

        # Centre de masse
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(result, (cx, cy), 10, (0, 0, 255), -1)

        # Ellipse
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            cv2.ellipse(result, ellipse, (255, 0, 0), 2)

        # Rectangle englobant
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 255), 2)

    return result

def pseudolandmarks(img, mask):
    """Détecte les pseudo-landmarks avec PlantCV"""
    # Trouve les contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = img.copy()

    if len(contours) > 0:
        # Prend le plus grand contour
        largest_contour = max(contours, key=cv2.contourArea)

        try:
            # Détecte les pseudo-landmarks avec PlantCV
            top, bottom, center_v = pcv.homology.x_axis_pseudolandmarks(mask=mask, img=img)
            left, right, center_h = pcv.homology.y_axis_pseudolandmarks(mask=mask, img=img)

            # Dessine les landmarks
            if top is not None and len(top) > 0:
                for point in top:
                    if len(point) >= 2:
                        cv2.circle(result, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)
            if bottom is not None and len(bottom) > 0:
                for point in bottom:
                    if len(point) >= 2:
                        cv2.circle(result, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)
            if left is not None and len(left) > 0:
                for point in left:
                    if len(point) >= 2:
                        cv2.circle(result, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
            if right is not None and len(right) > 0:
                for point in right:
                    if len(point) >= 2:
                        cv2.circle(result, (int(point[0]), int(point[1])), 5, (255, 255, 0), -1)
        except Exception as e:
            # Si PlantCV échoue, utilise un échantillonnage simple du contour
            num_landmarks = 20
            step = max(1, len(largest_contour) // num_landmarks)
            landmarks = largest_contour[::step]

            for point in landmarks:
                cv2.circle(result, tuple(point[0]), 5, (255, 0, 255), -1)

    return result

def color_histogram(img):
    """Crée un histogramme de couleurs avec matplotlib"""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    # Histogramme simple avec matplotlib
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
    """Traite une image et retourne/sauvegarde les transformations avec PlantCV"""
    # Paramètres PlantCV
    pcv.params.debug = None  # Options: 'print', 'plot', None

    # Lecture de l'image
    img, path, filename = pcv.readimage(filename=img_path)

    if img is None:
        print(f"Error: Cannot read image {img_path}")
        return None

    base_name = os.path.splitext(os.path.basename(img_path))[0]

    print(f"Processing {filename}...")

    # Génère toutes les transformations
    transformations = {}

    # 1. Original
    transformations['original'] = img

    # 2. Gaussian Blur
    print("  - Applying Gaussian blur...")
    transformations['gaussian_blur'] = gaussian_blur(img)

    # 3. Masque
    print("  - Creating mask...")
    mask = create_mask(img)
    transformations['mask'] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if len(mask.shape) == 2 else mask

    # 4. ROI Objects
    print("  - Detecting ROI objects...")
    transformations['roi_objects'] = roi_objects(img, mask)

    # 5. Analyze Object
    print("  - Analyzing object...")
    transformations['analyze_object'] = analyze_object(img, mask)

    # 6. Pseudolandmarks
    print("  - Extracting pseudolandmarks...")
    transformations['pseudolandmarks'] = pseudolandmarks(img, mask)

    # 7. Color Histogram
    print("  - Generating color histogram...")
    transformations['color_histogram'] = color_histogram(img)

    if save_dir:
        # Mode: sauvegarde dans un dossier
        os.makedirs(save_dir, exist_ok=True)
        for name, trans_img in transformations.items():
            if trans_img is not None:
                output_path = os.path.join(save_dir, f"{base_name}_{name}.jpg")
                pcv.print_image(img=trans_img, filename=output_path)
        print(f"  Transformations saved to {save_dir}")
    else:
        # Mode: affichage avec PlantCV
        print("  - Displaying results...")
        for name, trans_img in transformations.items():
            if trans_img is not None:
                pcv.plot_image(img=trans_img)

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
