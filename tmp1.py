import matplotlib.pyplot as plt
import numpy as np

# Generate random data
data = np.random.randn(1000)

# Plot histogram
plt.hist(data, bins=30, edgecolor='black')

# Add titles and labels
plt.title('Histogram of Data Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Show plot
plt.savefig('test.png')
