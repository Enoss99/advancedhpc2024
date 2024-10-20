import time
gaussian_kernel =[
    [0,0,1,2,1,0,0,],
    [0,3,13,22,13,3,0],
    [1,13,59,97,59,13,1],
    [2,22,97,159,97,22,2],
    [1,13,59,97,59,13,1],
    [0,3,13,22,13,3,0],
    [0,0,1,2,1,0,0,],
]
kernel_sum = sum([sum(row) for row in gaussian_kernel])
gaussian_kernel = [[element/kernel_sum for element in row] for row in gaussian_kernel]
print(f"Kernel sum: {kernel_sum}")

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

def gaussian_blur_not_shared(img):
    height, width = len(img), len(img[0])
    output_img = [[0 for _ in range(width)] for _ in range(height)]
    for x in range(3, width - 3):
        for y in range(3, height - 3):
            val = 0.0
            for i in range(-3, 4):
                for j in range(-3, 4):
                    val += img[y + j][x + i] * gaussian_kernel[j + 3][i + 3]
            output_img[y][x] = min(max(int(val), 0), 255)
    return output_img

def gaussian_blur_shared_memory(img):
    width, height = len(img[0]), len(img)
    output = [[0 for _ in range(width)] for _ in range(height)]
    
    for x in range(3, width - 3):
        for y in range(3, height - 3):
            shared_mem = [[img[y + j][x + i] for i in range(-3, 4)] for j in range(-3, 4)]
            val = 0.0
            for i in range(7):
                for j in range(7):
                    val += shared_mem[j][i] * gaussian_kernel[j][i]
            output[y][x] = int(val)
    
    return output

def measure_time(func, img):
    start = time.time()
    result = func(img)
    end = time.time()
    return result, end - start

#couldn't actually see the blurred image so i made this
def apply_multiple_passes(img, num_passes=5):
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
    blurred_img_no_shared, no_shared_time = measure_time(apply_multiple_passes, grayscale_img)
    print(f"Time without shared memory: {no_shared_time:.6f} seconds")
    blurred_img_shared, shared_time = measure_time(gaussian_blur_shared_memory, grayscale_img)
    print(f"Time with shared memory: {shared_time:.6f} seconds")
    speedup = no_shared_time / shared_time
    print(f"Speedup: {speedup:.2f}")

    save_image_ppm('blurred_image_no_shared.ppm', blurred_img_no_shared, width, height)
    save_image_ppm('blurred_image_shared.ppm', blurred_img_shared, width, height)