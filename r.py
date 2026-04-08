import torch
import time

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Something is wrong.")

device = torch.device("cuda")
print("Using GPU:", torch.cuda.get_device_name(0))

# Create large tensors to force GPU usage
# N = 16384
# x = torch.randn(N, N, device=device)
# w = torch.randn(N, N, device=device)

# # Warm-up
# _ = x @ w
# torch.cuda.synchronize()

# start = time.time()
# y = x @ w
# torch.cuda.synchronize()
# end = time.time()

# print("Matrix multiply done.")
# print("Output shape:", y.shape)
# print("Time taken (seconds):", end - start)
