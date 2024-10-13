import numba
from numba import cuda

def get_gpu_info():
    
    device = cuda.get_current_device()
    print(f"Device Name: {device.name.decode('utf-8')}")
    print(f"Multiprocessor Count: {device.MULTIPROCESSOR_COUNT}")

    # Memory info using current context
    mem_info = cuda.current_context().get_memory_info()
    print(f"Total Memory: {mem_info[1] / (1024 ** 3):.2f} GB")  # Total memory in GB
    print(f"Free Memory: {mem_info[0] / (1024 ** 3):.2f} GB")   # Free memory in GB

# Detect and list all available GPUs
def detect_gpus():
    numba.cuda.detect()

# Main execution
if __name__ == "__main__":
    # Optional: detect GPUs
    detect_gpus()
    # Select and print GPU info
    get_gpu_info()
