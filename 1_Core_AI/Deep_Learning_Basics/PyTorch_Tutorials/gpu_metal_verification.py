import tensorflow as tf
import torch
import platform

print("System Information:")
print(f"Python version: {platform.python_version()}")
print(f"Platform: {platform.platform()}")

print("\nTensorFlow Information:")
print(f"TensorFlow version: {tf.__version__}")
# Check for GPU (Metal) devices
physical_devices = tf.config.list_physical_devices()
print("Available TensorFlow devices:")
for device in physical_devices:
    print(f"- {device.device_type}: {device.name}")

print("\nPyTorch Information:")
print(f"PyTorch version: {torch.__version__}")
print("PyTorch MPS (Metal) available:", torch.backends.mps.is_available())
print("PyTorch MPS (Metal) built:", torch.backends.mps.is_built())


# Create and run a simple operation
with tf.device('/GPU:0'):
    a = tf.random.normal([1000, 1000])
    b = tf.random.normal([1000, 1000])
    c = tf.matmul(a, b)
    print("TensorFlow computation completed on:", c.device)

# Test PyTorch with Metal
import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")
    # Create and run a simple operation
    a = torch.randn(1000, 1000, device=device)
    b = torch.randn(1000, 1000, device=device)
    c = torch.matmul(a, b)
    print("PyTorch computation completed on:", c.device)