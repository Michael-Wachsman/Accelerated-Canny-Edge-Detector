from PIL import Image
import os, sys


input_image_path = "./256x256.jpg"
output_folder = "imgs/wScale_imgs"

def generate_doubled_images(input_image_path, output_folder):
    # Open the original image
    origImg = Image.open(input_image_path)
    prevImg = None
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    squared = 0

    scales = [1,2,4,8,16,32,64]
    for scale in scales:
        tiled_image = None
        # Handle the 1x case
        if prevImg is None:
            width, height = origImg.size
            tiled_image = Image.new('RGB', (width, height))
            tiled_image.paste(origImg, (0,0))
        else:
            prev_width, prev_height = prevImg.size
            if squared == 0:
                # Create a blank image with the new dimensions
                tiled_image = Image.new('RGB', (prev_width, prev_height*2))

                # Tile the original image to fill the new image
                for j in range(0, prev_height*2, prev_height):
                    tiled_image.paste(prevImg, (0, j))
            else:
                tiled_image = Image.new('RGB', (prev_width * 2, prev_height))

                for i in range(0, prev_width*2, prev_width):
                    tiled_image.paste(prevImg, (i, 0))

        output_path = os.path.join(output_folder, f"{scale}x.jpg")
        tiled_image.save(output_path)
        print(f"Saved: {output_path}")
        prevImg = tiled_image
        squared = 1 - squared

def generate_tiled_images(input_image_path, output_folder, max_scale=50):
    original_image = Image.open(input_image_path)
    original_width, original_height = original_image.size

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Generate and save images at different scales
    for scale in range(1, max_scale + 1):
        new_width = original_width * scale
        new_height = original_height * scale

        # Create a blank image with the new dimensions
        tiled_image = Image.new('RGB', (new_width, new_height))

        # Tile the original image to fill the new image
        for i in range(0, new_width, original_width):
            for j in range(0, new_height, original_height):
                tiled_image.paste(original_image, (i, j))

        # Save the new image
        output_path = os.path.join(output_folder, f"tiled_{new_width}x{new_height}.jpg")
        tiled_image.save(output_path)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    opt = sys.argv[1]
    if opt == 'weak':
        generate_doubled_images(input_image_path, output_folder)
    else:
        generate_tiled_images(input_image_path, output_folder)
