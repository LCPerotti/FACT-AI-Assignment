import pandas as pd
import matplotlib.pyplot as plt
import os

# Get the current directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Define the file path
file_path = os.path.join(current_directory, 'data experiment 2 figure 3.csv')

# Read the data
data = pd.read_csv(file_path)

# Filter data for attn_out and mlp_out
attn_out_data = data[data['label'].str.contains('attn_out')]
mlp_out_data = data[data['label'].str.contains('mlp_out')]

# Create subplots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

layers = [1,2,3,4,5,6,7,8,9,10,11,12]

# Plot attn_out data
axes[0].bar(attn_out_data['label'], attn_out_data['cp_mean - mem_mean'], color='orange', zorder=3)
axes[0].set_xlabel('Layers')
axes[0].set_ylabel(r'$\Delta_\text{cofa}$')
axes[0].set_title('Attention Block')
axes[0].tick_params(axis='x')

# Plot mlp_out data
axes[1].bar(mlp_out_data['label'], mlp_out_data['cp_mean - mem_mean'], color='purple', zorder=3)
axes[1].set_xlabel('Layers')
axes[1].set_ylabel(r'$\Delta t_\text{cofa}$')
axes[1].set_title('MLP Block')
axes[1].tick_params(axis='x')

# Adjust layout
plt.tight_layout()

# Set x-axis to show layer values and scale y-axis
for ax in axes:
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers)
    ax.set_ylim([-1, 1.5])
    ax.set_yticks([i * 0.5 for i in range(-2, 4)])  # y-axis ticks separated by 0.5
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.25))
    ax.grid(which='both', axis='y', linestyle='--', linewidth=0.5, zorder=0)

# Save the figure in the current directory
plt.savefig(os.path.join(current_directory, 'figure 3 recreated.png'))

plt.show()