import numpy as np
import pandas as pd

# Step 1: define a simple composition (3 parts)
composition = np.array([
    [0.2, 0.3, 0.5],
    [0.1, 0.1, 0.8],
    [0.4, 0.4, 0.2]
])
print("Original composition shape:", composition.shape)

# Step 2: apply clr transform manually
def clr(x):
    x = np.asarray(x)
    gm = np.exp(np.mean(np.log(x), axis=1))[:, np.newaxis]
    return np.log(x / gm)

clr_data = clr(composition)
print("\nCLR-transformed data:")
print(pd.DataFrame(clr_data))

# Check the shape
print("CLR data shape:", clr_data.shape)

# Step 3: check sum across parts (should be zero)
row_sums = np.sum(clr_data, axis=1)
print("\nSum across CLR components (should be zero):", row_sums)

# Step 4: compute rank (effective dimensionality)
rank = np.linalg.matrix_rank(clr_data)
print("Effective dimensionality (matrix rank):", rank)

# Step 5: show singular values (one should be zero)
u, s, vt = np.linalg.svd(clr_data, full_matrices=False)
print("\nSingular values:", s)
