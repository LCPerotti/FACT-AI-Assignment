import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Get the current directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Define the file path
file_path = os.path.join(current_directory, 'data experiment 3 figure 4a.csv')

# Read the data
data = pd.read_csv(file_path)

# Extract layer and head information
data['layer'] = data['label'].apply(lambda x: int(x.split('H')[0][1:]))
data['head'] = data['label'].apply(lambda x: int(x.split('H')[1]))

# Create a pivot table and transpose it
pivot_table = data.pivot_table(values='cp_mean - mem_mean', index='layer', columns='head').T

# Plot the heatmap using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(
    pivot_table,
    cmap='seismic',
    center=0,
    cbar_kws={'orientation': 'horizontal', 'shrink': 0.2},
    linewidths=0.5,
    linecolor='grey',
    square=True
)

# Reverse the y-axis
plt.gca().invert_yaxis()

# Customize the colorbar
cbar = plt.gcf().axes[-1]

cbar.xaxis.set_ticks_position('bottom')
cbar.xaxis.set_label_position('bottom')

# Move the colorbar label to the left and make it bigger
cbar.set_ylabel(r'$\Delta_\text{cofa}$', rotation=0, ha='right', fontsize=16)

# Adjust the position of the colorbar label
cbar.yaxis.set_label_coords(-0.1, -0.5)

# Adjust the position of the colorbar
plt.subplots_adjust(bottom=0.2)

# Customize the headers
plt.xlabel('Layer', fontsize=16)
plt.ylabel('Head', fontsize=16)

# Save the figure in the current directory
plt.savefig(os.path.join(current_directory, 'figure 4a recreated.png'))

# Show the plot
plt.show()
