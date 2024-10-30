from numba import cuda, float32
import math
import time

# Utility function to load PPM images
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

# Utility function to save PPM images
def save_image_ppm(filepath, img, width, height):
    with open(filepath, 'wb') as f:
        f.write(f"P6\n{width} {height}\n255\n".encode())
        for row in img:
            for pixel in row:
                r, g, b = pixel
                f.write(bytes([r, g, b]))

# Kuwahara filter without shared memory
@cuda.jit
def kuwahara_filter(image, output, window_size):
    x, y = cuda.grid(2)
    height = len(image)
    width = len(image[0])

    if x >= window_size and y >= window_size and x < height - window_size and y < width - window_size:
        best_mean_r = 0
        best_mean_g = 0
        best_mean_b = 0
        lowest_variance = math.inf

        for wx in range(2):
            for wy in range(2):
                mean_r = 0
                mean_g = 0
                mean_b = 0
                count = 0
                variance = 0

                for i in range(window_size):
                    for j in range(window_size):
                        xi = x + wx * window_size + i - window_size // 2
                        yj = y + wy * window_size + j - window_size // 2

                        r, g, b = image[xi][yj]
                        mean_r += r
                        mean_g += g
                        mean_b += b
                        count += 1

                mean_r /= count
                mean_g /= count
                mean_b /= count

                for i in range(window_size):
                    for j in range(window_size):
                        xi = x + wx * window_size + i - window_size // 2
                        yj = y + wy * window_size + j - window_size // 2

                        r, g, b = image[xi][yj]
                        variance += ((r - mean_r) ** 2 + (g - mean_g) ** 2 + (b - mean_b) ** 2) / 3

                variance /= count

                if variance < lowest_variance:
                    lowest_variance = variance
                    best_mean_r = mean_r
                    best_mean_g = mean_g
                    best_mean_b = mean_b

        output[x][y] = (int(best_mean_r), int(best_mean_g), int(best_mean_b))

# Kuwahara filter with shared memory
@cuda.jit
def kuwahara_filter_shared(image, output, window_size):
    x, y = cuda.grid(2)
    height = len(image)
    width = len(image[0])

    shared_tile = cuda.shared.array((32, 32, 3), dtype=float32)

    if x < height and y < width:
        shared_tile[cuda.threadIdx.x, cuda.threadIdx.y, 0] = image[x][y][0]
        shared_tile[cuda.threadIdx.x, cuda.threadIdx.y, 1] = image[x][y][1]
        shared_tile[cuda.threadIdx.x, cuda.threadIdx.y, 2] = image[x][y][2]
    cuda.syncthreads()

    if x >= window_size and y >= window_size and x < height - window_size and y < width - window_size:
        best_mean_r = 0
        best_mean_g = 0
        best_mean_b = 0
        lowest_variance = math.inf

        for wx in range(2):
            for wy in range(2):
                mean_r = 0
                mean_g = 0
                mean_b = 0
                count = 0
                variance = 0

                for i in range(window_size):
                    for j in range(window_size):
                        xi = x + wx * window_size + i - window_size // 2
                        yj = y + wy * window_size + j - window_size // 2

                        r = shared_tile[xi - x + window_size // 2, yj - y + window_size // 2, 0]
                        g = shared_tile[xi - x + window_size // 2, yj - y + window_size // 2, 1]
                        b = shared_tile[xi - x + window_size // 2, yj - y + window_size // 2, 2]

                        mean_r += r
                        mean_g += g
                        mean_b += b
                        count += 1

                mean_r /= count
                mean_g /= count
                mean_b /= count

                for i in range(window_size):
                    for j in range(window_size):
                        xi = x + wx * window_size + i - window_size // 2
                        yj = y + wy * window_size + j - window_size // 2

                        r = shared_tile[xi - x + window_size // 2, yj - y + window_size // 2, 0]
                        g = shared_tile[xi - x + window_size // 2, yj - y + window_size // 2, 1]
                        b = shared_tile[xi - x + window_size // 2, yj - y + window_size // 2, 2]

                        variance += ((r - mean_r) ** 2 + (g - mean_g) ** 2 + (b - mean_b) ** 2) / 3

                variance /= count

                if variance < lowest_variance:
                    lowest_variance = variance
                    best_mean_r = mean_r
                    best_mean_g = mean_g
                    best_mean_b = mean_b

        output[x][y] = (int(best_mean_r), int(best_mean_g), int(best_mean_b))

if __name__ == "__main__":
    img, width, height = loaf_image('drunkCat.ppm')
    window_size = 3  

    output_img = [[(0, 0, 0) for _ in range(width)] for _ in range(height)]
    d_image = cuda.to_device(img)
    d_output = cuda.to_device(output_img)
    threads_per_block = (16, 16)
    blocks_per_grid_x = (width + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (height + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    start = time.time()
    kuwahara_filter[blocks_per_grid, threads_per_block](d_image, d_output, window_size)
    output_img_non_shared = d_output.copy_to_host()
    non_shared_time = time.time() - start
    save_image_ppm("output_image_non_shared.ppm", output_img_non_shared, width, height)
    print(f"Non-shared memory execution time: {non_shared_time:.6f} seconds")

    start = time.time()
    kuwahara_filter_shared[blocks_per_grid, threads_per_block](d_image, d_output, window_size)
    output_img_shared = d_output.copy_to_host()
    shared_time = time.time() - start
    save_image_ppm("output_image_shared.ppm", output_img_shared, width, height)
    print(f"Shared memory execution time: {shared_time:.6f} seconds")
    print(f"Speedup: {non_shared_time / shared_time:.2f}")
