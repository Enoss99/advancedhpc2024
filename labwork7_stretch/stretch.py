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

def save_image_ppm(filepath, img, width, height):
    with open(filepath, 'wb') as f:
        f.write(f"P6\n{width} {height}\n255\n".encode())
        for row in img:
            for pixel in row:
                r = g = b = pixel
                f.write(bytes([r, g, b]))

@cuda.jit
def find_min_max(image, min_max):
    x, y = cuda.grid(2)
    if x < image.shape[0] and y < image.shape[1]:
        pixel_value = int(image[x, y])
        cuda.atomic.min(min_max, 0, pixel_value)
        cuda.atomic.max(min_max, 1, pixel_value)

@cuda.jit
def stretch_contrast(image, output, min_val, max_val):
    x, y = cuda.grid(2)
    if x < image.shape[0] and y < image.shape[1]:
        pixel_value = image[x, y]
        if max_val > min_val:
            stretched = (pixel_value - min_val) / (max_val - min_val) * 255
            output[x, y] = min(255, max(0, int(stretched)))
        else:
            output[x, y] = pixel_value

if __name__ == "__main__":
    img, width, height = loaf_image('peterAsylum.ppm')
    grayscale_img = rgb_to_grayscale(img)
    
    grayscale_array = np.ascontiguousarray(np.array(grayscale_img, dtype=np.uint8))

    threads_per_block = (16, 16)
    blocks_per_grid_x = (width + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (height + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    min_max = np.array([255, 0], dtype=np.int32)
    d_image = cuda.to_device(grayscale_array)
    d_min_max = cuda.to_device(min_max)
    find_min_max[blocks_per_grid, threads_per_block](d_image, d_min_max)
    min_max = d_min_max.copy_to_host()
    min_val, max_val = min_max[0], min_max[1]


    stretched_output = np.zeros((height, width), dtype=np.uint8)
    d_output = cuda.device_array_like(stretched_output)
    stretch_contrast[blocks_per_grid, threads_per_block](d_image, d_output, min_val, max_val)
    cuda.synchronize()
    stretched_output = d_output.copy_to_host()

    save_image_ppm("stretched_image.ppm", stretched_output.tolist(), width, height)
