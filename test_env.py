import os
import sys
import torch
import cv2
import numpy as np

print("Python:", sys.version)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)

print("\nWorkspace files:")
print(os.listdir("/workspace"))

print("\nData root:")
print(os.listdir("/data"))

if os.path.exists("/data/Data"):
    print("\n/data/Data:")
    print(os.listdir("/data/Data")[:20])
