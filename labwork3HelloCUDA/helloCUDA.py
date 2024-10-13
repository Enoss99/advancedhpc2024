import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from numba import cuda, jit
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

def rgb_to_grayscale_cpu(flat_img):
    grayscale_img = np.dot(flat_img[...,:3], [0.2989, 0.5870, 0.1140])
    return grayscale_img

@cuda.jit
def rgb_to_grayscale_gpu_kernel(flat_img, grayscale_img):
    idx = cuda.grid(1)
    if idx < flat_img.shape[0]:
        r = flat_img[idx, 0]
        g = flat_img[idx, 1]
        b = flat_img[idx, 2]
        grayscale_img[idx] = 0.2989 * r + 0.5870 * g + 0.1140 * b

def rgb_to_grayscale_gpu(flat_img, threads_per_block=256):
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
    return grayscale_img

if __name__ == "__main__":
    img = load_image('drunkCat.jpg')
    print(f"Image size: {img.size}")
    flat_img = flatten_image(img)
    
    #CPU
    start_time = time.time()
    grayscale_img_cpu = rgb_to_grayscale_cpu(flat_img)
    cpu_time = time.time() - start_time
    print(f"CPU time: {cpu_time:.6f} seconds")    
    grayscale_img_cpu = grayscale_img_cpu.reshape(img.shape[0], img.shape[1])

    plt.imshow(grayscale_img_cpu, cmap='gray')
    plt.show()

    #GPU
    start_time = time.time()
    grayscale_img_gpu = rgb_to_grayscale_gpu(flat_img)
    gpu_time = time.time() - start_time
    print(f"GPU time: {gpu_time:.6f} seconds")
    grayscale_img_gpu = grayscale_img_gpu.reshape(img.shape[0], img.shape[1])

    plt.imsave('grayscaled_drunkCat.jpg', grayscale_img_gpu, cmap='gray')

    plt.imshow(grayscale_img_gpu, cmap='gray')
    plt.show()

    # GPU implementation time with != block sizes
    block_sizes = [32, 64, 128, 256, 512, 1024]
    gpu_times = []

    for block_size in block_sizes:
        start_time = time.time()
        grayscale_img_gpu = rgb_to_grayscale_gpu(flat_img, block_size)
        gpu_time = time.time() - start_time
        gpu_times.append(gpu_time)
        print(f"GPU time with block size {block_size}: {gpu_time:.6f} seconds")

    with open('gpu_times.txt', 'w') as f:
        for block_size, gpu_time in zip(block_sizes, gpu_times):
            f.write(f"{block_size} {gpu_time}\n")

    plt.plot(block_sizes, gpu_times, marker='o')
    plt.xlabel('Block Size')
    plt.ylabel('Time (seconds)')
    plt.title('Block Size vs Time')
    plt.grid(True)
    plt.savefig('block_size_vs_time.png')
    plt.show()

    speedup = cpu_time / gpu_time
    print(f"Speedup: {speedup:.2f}x")