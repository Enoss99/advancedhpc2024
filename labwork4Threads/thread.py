import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from numba import cuda
import time

def load_image(image_path):
    img = plt.imread(image_path)
    return img

def flatten_image(img):
    #input (h, w, 3)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    pixel_count = img.shape[0] * img.shape[1]
    print(f"Image shape: {img.shape}")
    print(f"Calculated pixel count: {pixel_count}")
    flat_img = img.reshape(pixel_count, 3)
    return flat_img

def rgb_to_grayscale_cpu(img):
    grayscale_img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    return grayscale_img
'''@cuda.jit
def rgb_to_grayscale_gpu_kernel(flat_img, grayscale_img):
    idx = cuda.grid(1)
    if idx < flat_img.shape[0]:
        r = flat_img[idx, 0]
        g = flat_img[idx, 1]
        b = flat_img[idx, 2]
        grayscale_img[idx] = 0.2989 * r + 0.5870 * g + 0.1140 * b
'''
@cuda.jit
def rgb_to_grayscale_gpu_kernel_2d(img, grayscale_img):
    x, y = cuda.grid(2)
    if x < img.shape[0] and y < img.shape[1]:
        r, g, b = img[x, y, 0], img[x, y, 1], img[x, y, 2]
        grayscale_img[x, y] = 0.2989 * r + 0.5870 * g + 0.1140 * b

'''def rgb_to_grayscale_gpu(flat_img, threads_per_block=256):
    # array contiguous
    flat_img = np.ascontiguousarray(flat_img)
    
    # feed data
    flat_img_device = cuda.to_device(flat_img)
    grayscale_img_device = cuda.device_array(flat_img.shape[0], dtype=np.float32)
    
    # launch the kernel
    blocks_per_grid = (flat_img.shape[0] + (threads_per_block - 1)) // threads_per_block
    rgb_to_grayscale_gpu_kernel[blocks_per_grid, threads_per_block](flat_img_device, grayscale_img_device)
    
    #result
    grayscale_img = grayscale_img_device.copy_to_host()
    return grayscale_img'''
def rgb_to_grayscale_gpu_2d(img, block_size=(16, 16)):
    img = np.ascontiguousarray(img)
    
    grayscale_img_device = cuda.device_array((img.shape[0], img.shape[1]), dtype=np.float32)
    
    img_device = cuda.to_device(img)
    
    threads_per_block = block_size
    blocks_per_grid_x = int(np.ceil(img.shape[0] / threads_per_block[0]))
    blocks_per_grid_y = int(np.ceil(img.shape[1] / threads_per_block[1]))
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    
    rgb_to_grayscale_gpu_kernel_2d[blocks_per_grid, threads_per_block](img_device, grayscale_img_device)
    
    grayscale_img = grayscale_img_device.copy_to_host()
    
    return grayscale_img

if __name__ == "__main__":
    img = load_image('drunkCat.jpg')
    
    # CPU
    '''
    start_time = time.time()
    grayscale_img_cpu = rgb_to_grayscale_cpu(img)
    cpu_time = time.time() - start_time
    print(f"CPU time: {cpu_time:.6f} seconds")
    
    plt.imshow(grayscale_img_cpu, cmap='gray')
    plt.show()
'''
    # GPU 2D blocks
    block_sizes = [(8, 8), (16, 16), (32, 32)]
    gpu_times = []

    for block_size in block_sizes:
        start_time = time.time()
        grayscale_img_gpu = rgb_to_grayscale_gpu_2d(img, block_size)
        gpu_time = time.time() - start_time
        gpu_times.append(gpu_time)
        print(f"GPU time with block size {block_size}: {gpu_time:.6f} seconds")
        
        plt.imshow(grayscale_img_gpu, cmap='gray')
        plt.show()
    

    speedups = [gpu_time for gpu_time in gpu_times]

    block_sizes_flat = [block_size[0] for block_size in block_sizes]
    plt.plot(block_sizes_flat, speedups, marker='o')
    plt.xlabel('Block Size (1D)')
    plt.ylabel('Speedup')
    plt.title('Block Size vs Speedup (2D Blocks)')
    plt.grid(True)
    plt.savefig('block_size_vs_speedup.png')
    plt.show()
    
    print(f"Speedup: {speedups}")