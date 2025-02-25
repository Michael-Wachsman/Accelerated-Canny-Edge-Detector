from PIL import Image
import numpy as np
import sys


def compare_images(image_path1, image_path2):
    try:
        img1 = Image.open(image_path1).convert("RGB")  # Convert to RGB to handle differences in mode
        img2 = Image.open(image_path2).convert("RGB")
        
        if img1.size != img2.size:
            raise ValueError("Images do not have the same dimensions. Cannot compare.")

        np_img1 = np.array(img1)
        np_img2 = np.array(img2)

        num_pixels = img1.size[0] * img1.size[1]

        diff = (np_img1 != np_img2)
        # print(diff)
        diff_pixels = np.any(diff, axis=2)  # Check if any channel differs
        # print(diff_pixels)
        highlight_img = np.zeros_like(np_img1)
        
        highlight_img[diff_pixels] = [255, 255, 255]
        highlighted_img = Image.fromarray(highlight_img)
        output_path = './diff.jpg'
        highlighted_img.save(output_path)

        num_diff = np.sum(diff_pixels)
        num_diff = max(0, num_diff)
        print(f"Number of differing pixels: {num_diff}")
        print(f"Percentage difference: {num_diff / num_pixels * 100}%")
    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python correctness.py <image1_path> <image2_path>")
        sys.exit(1)

    # Get image paths from command line
    image1_path = sys.argv[1]
    image2_path = sys.argv[2]
    compare_images(image1_path, image2_path)
