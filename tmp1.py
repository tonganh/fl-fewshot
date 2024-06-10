import matplotlib.pyplot as plt

# Sample data
categories = ['A', 'B', 'C', 'D']
values1 = [20, 30, 25, 35]
values2 = [15, 25, 20, 30]

# Plotting
plt.bar(categories, values1, label='Group 1')
plt.bar(categories, values2, bottom=values1, label='Group 2')

# Adding labels and title
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Stacked Bar Plot')
plt.legend()

# Display the plot
plt.savefig('test.png')

