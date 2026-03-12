# Create array of 20 random integers and find min, max, mean, sum

import numpy as np

random_array = np.random.randint(1, 100, 20)

print("Array:", random_array)
print("Minimum:", random_array.min())
print("Maximum:", random_array.max())
print("Mean:", random_array.mean())
print("Sum:", random_array.sum())
