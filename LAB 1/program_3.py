# Create NumPy array from 1 to 12 and reshape into 3×4

import numpy as np

original_array = np.arange(1, 13)
reshaped_array = original_array.reshape(3, 4)

print("Original Array:", original_array)
print("Original Shape:", original_array.shape)

print("\nReshaped Array:")
print(reshaped_array)
print("Reshaped Shape:", reshaped_array.shape)
