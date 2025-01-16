import pandas as pd
import matplotlib.pyplot as plt
import os

# Get the current directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Define the file path
file_path = os.path.join(current_directory, 'data experiment 1 figure 2b.csv')

# Read the data
data = pd.read_csv(file_path)

# Extract the data for plotting
layers = data['layer']
mem_values = data['mem']
cp_values = data['cp']

# Create the plot
plt.figure(figsize=(8, 6))

# Plot 'mem' values in blue with thicker lines
plt.plot(layers, mem_values, label='Factual Token', marker='o', color='blue', linewidth=3)

# Plot 'cp' values in red with thicker lines
plt.plot(layers, cp_values, label='Counterfactual token', marker='o', color='red', linewidth=3)

# Add labels and title
plt.xlabel('Layer', fontsize=18)
plt.ylabel('Logits in the Last Position', fontsize=18)

# Set y-axis tick marks
plt.yticks([0, 5, 10, 15])

# Place the legend underneath the plot area but inside the figure
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fontsize=12, ncol=2)

# Adjust layout to add padding
plt.tight_layout(pad=1.0)

# Save the figure in the current directory
plt.savefig(os.path.join(current_directory, 'figure 2b recreated.png'), bbox_inches='tight')

# Show the plot
plt.show()