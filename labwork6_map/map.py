from numba import cuda
import numpy as np

def loaf_image(image_path):
    with open(image_path, 'rb') as f:
        header = f.readline().decode().strip()
        if header != 'P6':
            raise ValueError(f"Unsupported PPM format: {header}")
        dimensions = f.readline().decode().strip()
        while dimensions.startswith('#'):
            dimensions = f.readline().decode().strip()
        width, height = map(int, dimensions.split())
        max_val = f.readline().decode().strip()
        if max_val != '255':
            raise ValueError(f"Unexpected max color value: {max_val}")
        img_data = f.read()
        img = []
        for y in range(height):
            row = []
            for x in range(width):
                idx = (y * width + x) * 3
                r = img_data[idx]
                g = img_data[idx + 1]
                b = img_data[idx + 2]
                row.append((r, g, b))
            img.append(row)
    return img, width, height

def rgb_to_grayscale(img):
    height, width = len(img), len(img[0])
    grayscale_img = [[0 for _ in range(width)] for _ in range(height)]

    for y in range(height):
        for x in range(width):
            r, g, b = img[y][x]
            grayscale_value = 0.2989 * r + 0.5870 * g + 0.1140 * b
            grayscale_img[y][x] = int(grayscale_value)

    return grayscale_img

@cuda.jit
def binarize_image(image, output, threshold):
    x, y = cuda.grid(2)
    if x < len(image) and y < len(image[0]):  
        pixel_value = image[x][y]
        output[x][y] = 255 if pixel_value >= threshold else 0

@cuda.jit
def adjust_brightness(image, output, brightness_offset):
    x, y = cuda.grid(2)
    if x < len(image) and y < len(image[0]):  
        new_value = image[x][y] + brightness_offset
        output[x][y] = min(255, max(0, new_value))

@cuda.jit
def blend_images(image1, image2, output, blend_coef):
    x, y = cuda.grid(2)
    if x < len(image1) and y < len(image1[0]):
        blended_value = blend_coef * image1[x][y] + (1 - blend_coef) * image2[x][y]
        output[x][y] = min(255, max(0, int(blended_value)))

def save_image_ppm(filepath, img, width, height):
    with open(filepath, 'wb') as f:

        f.write(f"P6\n{width} {height}\n255\n".encode())
        for row in img:
            for pixel in row:
                r = g = b = pixel
                f.write(bytes([r, g, b]))

if __name__ == "__main__":
    img, width, height = loaf_image('drunkCat.ppm')
    grayscale_img = rgb_to_grayscale(img)

    d_input = cuda.to_device(np.array(grayscale_img, dtype=np.uint8))
    d_output = cuda.device_array_like(d_input)
    threads_per_block = (16, 16)
    blocks_per_grid_x = (width + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (height + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    threshold = 128
    binarize_image[blocks_per_grid, threads_per_block](d_input, d_output, threshold)
    binarized_output = d_output.copy_to_host()
    save_image_ppm("binarized_image.ppm", binarized_output, width, height)

    brightness_offset = 50
    adjust_brightness[blocks_per_grid, threads_per_block](d_input, d_output, brightness_offset)
    brightness_output = d_output.copy_to_host()
    save_image_ppm("adjusted_brightness_image.ppm", brightness_output, width, height)

    img2 = grayscale_img[::-1]
    d_image2 = cuda.to_device(img2)
    d_output = cuda.device_array((height, width), dtype=int)
    blend_images[blocks_per_grid, threads_per_block](d_input, d_image2, d_output, 0.5)
    blended_output = d_output.copy_to_host()
    save_image_ppm("blended_image.ppm", blended_output, width, height)
