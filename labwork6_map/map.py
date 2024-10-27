import numpy as np
from numba import cuda

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
    return np.array(img, dtype=np.uint8), width, height

def rgb_to_grayscale(img):
    height, width = img.shape[0], img.shape[1]
    grayscale_img = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            r, g, b = img[y, x]
            grayscale_img[y, x] = int(0.2989 * r + 0.5870 * g + 0.1140 * b)
    return grayscale_img

@cuda.jit
def binarize_image(image, output, threshold):
    x, y = cuda.grid(2)
    if x < image.shape[0] and y < image.shape[1]:  
        pixel_value = image[x, y]
        output[x, y] = 1 if pixel_value >= threshold else 0

@cuda.jit
def adjust_brightness(image, output, brightness_offset):
    x, y = cuda.grid(2)
    if x < image.shape[0] and y < image.shape[1]:  
        new_value = image[x, y] + brightness_offset
        output[x, y] = max(0, min(255, new_value))

@cuda.jit
def blend_images(image1, image2, output, blend_coef):
    x, y = cuda.grid(2)
    if x < image1.shape[0] and y < image1.shape[1]:
        blended_value = blend_coef * image1[x, y] + (1 - blend_coef) * image2[x, y]
        output[x, y] = int(min(255, max(0, blended_value)))  

if __name__ == "__main__":
    img, width, height = loaf_image('drunkCat.ppm')
    grayscale_img = rgb_to_grayscale(img)  
    
    
    threads_per_block = (16, 16)
    blocks_per_grid_x = (width + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (height + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    
    threshold = np.uint8(128) 
    binarized_output = np.zeros((height, width), dtype=np.uint8)
    d_image = cuda.to_device(grayscale_img) 
    d_output = cuda.device_array_like(binarized_output)
    binarize_image[blocks_per_grid, threads_per_block](d_image, d_output, threshold)
    binarized_output = d_output.copy_to_host() * 255  

    with open("binarized_image.pgm", "wb") as f: 
        f.write(b"P5\n")
        f.write(f"{width} {height}\n".encode())
        f.write(b"255\n")
        f.write(binarized_output.tobytes())

    
    brightness_offset = 50
    brightness_output = np.zeros((height, width), dtype=np.uint8)
    adjust_brightness[blocks_per_grid, threads_per_block](d_image, d_output, brightness_offset)
    brightness_output = d_output.copy_to_host()

    with open("adjusted_brightness_image.pgm", "wb") as f:  
        f.write(b"P5\n")
        f.write(f"{width} {height}\n".encode())
        f.write(b"255\n")
        f.write(brightness_output.tobytes())

   
    img2 = np.ascontiguousarray(np.flip(grayscale_img, axis=0))  
    d_image2 = cuda.to_device(img2) 
    blended_output = np.zeros((height, width), dtype=np.uint8)
    d_output = cuda.device_array_like(blended_output)
    blend_images[blocks_per_grid, threads_per_block](d_image, d_image2, d_output, 0.5)
    blended_output = d_output.copy_to_host()

    with open("blended_image.pgm", "wb") as f:  
        f.write(b"P5\n")
        f.write(f"{width} {height}\n".encode())
        f.write(b"255\n")
        f.write(blended_output.tobytes())
