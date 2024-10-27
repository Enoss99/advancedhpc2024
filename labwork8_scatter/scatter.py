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

def save_image_ppm(filepath, img, width, height):
    with open(filepath, 'wb') as f:
        f.write(f"P6\n{width} {height}\n255\n".encode())
        for row in img:
            for pixel in row:
                r, g, b = pixel
                f.write(bytes([r, g, b]))

@cuda.jit
def rgb_to_hsv_kernel(r, g, b, h, s, v):
    x, y = cuda.grid(2)
    if x < r.shape[0] and y < r.shape[1]:
        rf, gf, bf = r[x, y] / 255.0, g[x, y] / 255.0, b[x, y] / 255.0
        cmax = max(rf, gf, bf)
        cmin = min(rf, gf, bf)
        delta = cmax - cmin

        if delta == 0:
            h[x, y] = 0
        elif cmax == rf:
            h[x, y] = (60 * ((gf - bf) / delta) + 360) % 360
        elif cmax == gf:
            h[x, y] = (60 * ((bf - rf) / delta) + 120) % 360
        else:
            h[x, y] = (60 * ((rf - gf) / delta) + 240) % 360

        s[x, y] = 0 if cmax == 0 else (delta / cmax)

        v[x, y] = cmax

@cuda.jit
def hsv_to_rgb_kernel(h, s, v, r, g, b):
    x, y = cuda.grid(2)
    if x < h.shape[0] and y < h.shape[1]:
        hh = h[x, y] / 60.0
        i = int(hh) % 6
        f = hh - i
        p = v[x, y] * (1 - s[x, y])
        q = v[x, y] * (1 - f * s[x, y])
        t = v[x, y] * (1 - (1 - f) * s[x, y])

        if i == 0:
            rf, gf, bf = v[x, y], t, p
        elif i == 1:
            rf, gf, bf = q, v[x, y], p
        elif i == 2:
            rf, gf, bf = p, v[x, y], t
        elif i == 3:
            rf, gf, bf = p, q, v[x, y]
        elif i == 4:
            rf, gf, bf = t, p, v[x, y]
        else:
            rf, gf, bf = v[x, y], p, q

        r[x, y] = int(rf * 255)
        g[x, y] = int(gf * 255)
        b[x, y] = int(bf * 255)

if __name__ == "__main__":
    img, width, height = loaf_image("drunkCat.ppm")
    r = np.array([[px[0] for px in row] for row in img], dtype=np.uint8)
    g = np.array([[px[1] for px in row] for row in img], dtype=np.uint8)
    b = np.array([[px[2] for px in row] for row in img], dtype=np.uint8)

    h = np.zeros((height, width), dtype=np.float32)
    s = np.zeros((height, width), dtype=np.float32)
    v = np.zeros((height, width), dtype=np.float32)

    threads_per_block = (16, 16)
    blocks_per_grid_x = (width + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (height + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    rgb_to_hsv_kernel[blocks_per_grid, threads_per_block](r, g, b, h, s, v)
    cuda.synchronize()

    r_out = np.zeros((height, width), dtype=np.uint8)
    g_out = np.zeros((height, width), dtype=np.uint8)
    b_out = np.zeros((height, width), dtype=np.uint8)

    hsv_to_rgb_kernel[blocks_per_grid, threads_per_block](h, s, v, r_out, g_out, b_out)
    cuda.synchronize()

    final_img = [[(r_out[y, x], g_out[y, x], b_out[y, x]) for x in range(width)] for y in range(height)]

    save_image_ppm("reconstructed_image.ppm", final_img, width, height)
