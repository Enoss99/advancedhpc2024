import time
from numba import cuda, float32
import numpy as np

gaussian_kernel = np.array([
    [0, 0, 1, 2, 1, 0, 0],
    [0, 3, 13, 22, 13, 3, 0],
    [1, 13, 59, 97, 59, 13, 1],
    [2, 22, 97, 159, 97, 22, 2],
    [1, 13, 59, 97, 59, 13, 1],
    [0, 3, 13, 22, 13, 3, 0],
    [0, 0, 1, 2, 1, 0, 0]
], dtype=np.float32)
kernel_sum = np.sum(gaussian_kernel)
gaussian_kernel = gaussian_kernel / kernel_sum  

def gaussian_blur_not_shared(img):
    height, width = len(img), len(img[0])
    output_img = [[0 for _ in range(width)] for _ in range(height)]
    for x in range(3, width - 3):
        for y in range(3, height - 3):
            val = 0.0
            for i in range(-3, 4):
                for j in range(-3, 4):
                    val += img[y + j][x + i] * gaussian_kernel[j + 3, i + 3]
            output_img[y][x] = min(max(int(val), 0), 255)
    return output_img

@cuda.jit
def gaussian_blur_shared_memory_cuda(input_img, output_img, kernel):
    x, y = cuda.grid(2)
    if x < 3 or y < 3 or x >= input_img.shape[0] - 3 or y >= input_img.shape[1] - 3:
        return 

    val = 0.0
    for i in range(-3, 4):
        for j in range(-3, 4):
            val += input_img[x + i, y + j] * kernel[i + 3, j + 3]
    output_img[x, y] = min(max(int(val), 0), 255)

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

def measure_time(func, img):
    start = time.time()
    result = func(img)
    end = time.time()
    return result, end - start

#couldn't actually see the blurred image so i made this
def apply_multiple_passes_NS(img, num_passes=5):
    blurred_img = img
    for _ in range(num_passes):
        blurred_img = gaussian_blur_not_shared(blurred_img)
    return blurred_img

def save_image_ppm(filepath, img, width, height):
    with open(filepath, 'wb') as f:

        f.write(f"P6\n{width} {height}\n255\n".encode())
        for row in img:
            for pixel in row:
                r = g = b = pixel
                f.write(bytes([r, g, b]))

if __name__ == "__main__":
    img, width, height = loaf_image('peterAsylum.ppm')
    grayscale_img = rgb_to_grayscale(img)
    #CPU
    blurred_img_no_shared = gaussian_blur_not_shared(grayscale_img)
    blurred_img_no_shared, no_shared_time = measure_time(gaussian_blur_not_shared, grayscale_img)
    print(f"Time without shared memory: {no_shared_time:.6f} seconds")
    #GPU
    d_input = cuda.to_device(np.array(grayscale_img, dtype=np.uint8))
    d_output = cuda.device_array_like(d_input)
    d_gaussian_kernel = cuda.to_device(gaussian_kernel)  
    threads_per_block = (16, 16)
    blocks_per_grid_x = (d_input.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (d_input.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    start = time.time()
    blurred_img_shared = gaussian_blur_shared_memory_cuda[blocks_per_grid, threads_per_block](d_input, d_output, d_gaussian_kernel)

    cuda.synchronize()
    shared_time = time.time() - start
    blurred_img_shared = d_output.copy_to_host()

    print(f"Time with shared memory: {shared_time:.6f} seconds")
    speedup = no_shared_time / shared_time
    print(f"Speedup: {speedup:.2f}")

    save_image_ppm('blurred_image_no_shared.ppm', blurred_img_no_shared, width, height)
    save_image_ppm('blurred_image_shared.ppm', blurred_img_shared, width, height)
