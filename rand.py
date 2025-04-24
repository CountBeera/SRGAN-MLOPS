import torch

def check_gpu():
    print("PyTorch version:", torch.__version__)

    # Check if CUDA is available
    if torch.cuda.is_available():
        print("✅ CUDA is available!")
        print("GPU Name:", torch.cuda.get_device_name(0))
        print("Total GPU Memory (MB):", torch.cuda.get_device_properties(0).total_memory // (1024 ** 2))
        print("Current Device:", torch.cuda.current_device())
        
        # Try allocating a tensor on the GPU
        try:
            x = torch.randn(3, 3).to("cuda")
            print("✅ Tensor successfully moved to GPU:")
            print(x)
        except Exception as e:
            print("❌ Failed to move tensor to GPU:", e)
    else:
        print("❌ CUDA is NOT available. Running on CPU.")

if __name__ == "__main__":
    
    check_gpu()
