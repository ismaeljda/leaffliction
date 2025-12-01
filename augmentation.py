import cv2
import numpy as np
import os
import random

def augment_image(img_path, output_dir):
    """
    Augmente une image avec 9 techniques:
    1. Flip horizontal
    2. Rotate (90°)
    3. Skew
    4. Shear
    5. Crop
    6. Distortion
    7. Blur
    8. Contrast
    9. Scale
    Sauvegarde les images augmentées avec le nom original + type d'augmentation.
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error reading {img_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(img_path))[0]
    ext = os.path.splitext(img_path)[1]

    h, w = img.shape[:2]
    augmented_images = []

    # 1. FLIP
    flipped = cv2.flip(img, 1)
    flip_name = os.path.join(output_dir, f"{base_name}_flip{ext}")
    cv2.imwrite(flip_name, flipped)
    augmented_images.append(flip_name)

    # 2. ROTATE
    angle = random.uniform(15, 90)
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    rotate_name = os.path.join(output_dir, f"{base_name}_rotate{ext}")
    cv2.imwrite(rotate_name, rotated)
    augmented_images.append(rotate_name)

    # 3. SKEW
    skew_amount = random.uniform(0.05, 0.15)
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    pts2 = np.float32([[0, h*skew_amount], [w, 0], [0, h], [w, h*(1-skew_amount)]])
    M_skew = cv2.getPerspectiveTransform(pts1, pts2)
    skewed = cv2.warpPerspective(img, M_skew, (w, h))
    skew_name = os.path.join(output_dir, f"{base_name}_skew{ext}")
    cv2.imwrite(skew_name, skewed)
    augmented_images.append(skew_name)

    # 4. SHEAR
    shear_x = random.uniform(0.1, 0.3)
    shear_y = random.uniform(0.1, 0.3)
    M_shear = np.float32([[1, shear_x, 0], [shear_y, 1, 0]])
    sheared = cv2.warpAffine(img, M_shear, (w, h))
    shear_name = os.path.join(output_dir, f"{base_name}_shear{ext}")
    cv2.imwrite(shear_name, sheared)
    augmented_images.append(shear_name)

    # 5. CROP (crop central avec marge aléatoire puis resize)
    crop_margin = int(min(h, w) * random.uniform(0.05, 0.15))
    cropped = img[crop_margin:h-crop_margin, crop_margin:w-crop_margin]
    cropped = cv2.resize(cropped, (w, h))
    crop_name = os.path.join(output_dir, f"{base_name}_crop{ext}")
    cv2.imwrite(crop_name, cropped)
    augmented_images.append(crop_name)

    # 6. DISTORTION (barrel distortion)
    k = random.uniform(0.2, 0.4)
    distorted = barrel_distortion(img, k)
    distortion_name = os.path.join(output_dir, f"{base_name}_distortion{ext}")
    cv2.imwrite(distortion_name, distorted)
    augmented_images.append(distortion_name)

    # 7. BLUR
    kernel_size = random.choice([7, 9, 11, 13, 15])
    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    blur_name = os.path.join(output_dir, f"{base_name}_blur{ext}")
    cv2.imwrite(blur_name, blurred)
    augmented_images.append(blur_name)

    # 8. CONTRAST
    alpha = random.uniform(1.2, 1.8)
    contrasted = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
    contrast_name = os.path.join(output_dir, f"{base_name}_contrast{ext}")
    cv2.imwrite(contrast_name, contrasted)
    augmented_images.append(contrast_name)

    # 9. SCALE (zoom aléatoire puis crop au centre)
    scale_factor = random.uniform(1.1, 1.3)
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    scaled = cv2.resize(img, (new_w, new_h))
    start_x = (new_w - w) // 2
    start_y = (new_h - h) // 2
    scaled = scaled[start_y:start_y+h, start_x:start_x+w]
    scale_name = os.path.join(output_dir, f"{base_name}_scale{ext}")
    cv2.imwrite(scale_name, scaled)
    augmented_images.append(scale_name)

    return augmented_images

def barrel_distortion(img, k):
    """Applique une distortion barrel (fish-eye)"""
    h, w = img.shape[:2]
    cx, cy = w/2, h/2

    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)

    for y in range(h):
        for x in range(w):
            dx = (x - cx) / cx
            dy = (y - cy) / cy
            r2 = dx*dx + dy*dy
            r4 = r2 * r2

            distortion = 1 + k * r2 + k * r4

            map_x[y, x] = cx + dx * distortion * cx
            map_y[y, x] = cy + dy * distortion * cy

    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

# -----------------------------
# Main test
# -----------------------------
if __name__ == "__main__":
    input_image = r"Apple\Apple_Black_rot\image (2).JPG"
    output_folder = r"augmented_images"

    augmented_files = augment_image(input_image, output_folder)
    print("Augmented images saved:")
    for f in augmented_files:
        print(f)
