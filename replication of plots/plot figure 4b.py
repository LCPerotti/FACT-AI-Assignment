import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Get the current directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Define the file path
file_path = os.path.join(current_directory, 'data experiment 3 figure 4b.csv')

# Read the data
data = pd.read_csv(file_path)

LayerHead = [(9,6),(9,9),(10,0),(10,7),(10,10),(11,10)]



##################################################################
#                    Only Source Position 13                     #
##################################################################

# Initialize an empty DataFrame to store the combined data
combined_data = pd.DataFrame()

# Loop through each layer, head pair and concatenate the filtered data
for layer, head in LayerHead:
    filtered_data = data[(data['layer'] == layer) & (data['head'] == head) & (data['source_position'] == 13)]
    filtered_data['layer_head'] = f'Layer {layer} | Head {head}'

    # Change the sign for all values with layer 10 and head 7
    if (layer,head) == (10,7) or (layer,head) == (11,10):
        filtered_data['value'] = -filtered_data['value']

    combined_data = pd.concat([combined_data, filtered_data])

# Set the order of the 'layer_head' category in reverse
combined_data['layer_head'] = pd.Categorical(combined_data['layer_head'], 
                                             categories=[f'Layer {layer} | Head {head}' for layer, head in reversed(LayerHead)], 
                                             ordered=True)

# Pivot the table to get the destination positions as columns
pivot_table_combined = combined_data.pivot(index='layer_head', columns='dest_position', values='value')


# Create colormaps for negative and positive values
cmap_neg = sns.color_palette("Blues", as_cmap=True)
cmap_pos = sns.color_palette("Reds", as_cmap=True)

# Calculate the absolute maximum value from the data
max_abs_value = max(abs(pivot_table_combined.min().min()), abs(pivot_table_combined.max().max()))

# Normalize both color bars to the same range
norm = mpl.colors.Normalize(vmin=-max_abs_value, vmax=max_abs_value)

# Create the main heatmap (plot stays unchanged)
plt.figure(figsize=(12, 8))
sns.heatmap(
    pivot_table_combined,
    cmap='seismic',  # Original heatmap remains unchanged
    center=0,  # Center the colormap at 0
    cbar=False,  # Disable the default color bar
    linewidths=0.1,
    linecolor='grey',
    square=True
)

# Set the x-axis label
plt.xlabel('Token Position', fontsize=12)

# Create a divider for the axes
ax = plt.gca()
divider = make_axes_locatable(ax)

# Add the first color bar (negative values - blue)
cax_neg = divider.append_axes("right", size="5%", pad=0.1)  # Bottom bar
cbar_neg = plt.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap_neg),
    cax=cax_neg,
    orientation='vertical',
)
# Set the label at the top with a multi-line format
cax_neg.set_title("Factual\nAttention", loc='left', fontsize=10)

# Add the second color bar (positive values - red) above the first
cax_pos = divider.append_axes("right", size="5%", pad=0.8)  # Top bar
cbar_pos = plt.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap_pos),
    cax=cax_pos,
    orientation='vertical',
)
# Set the label at the top with a multi-line format
cax_pos.set_title("Counterfactual\nAttention", loc='left', fontsize=10)

# Save the figure in the current directory
plt.savefig(os.path.join(current_directory, 'figure 4b recreated.png'))

# Show the plot
plt.show()



#################################################################
#                     Every Source Position                     #
#################################################################

# Create a figure with 14 subplots stacked vertically
fig, axes = plt.subplots(nrows=14, ncols=1, figsize=(12, 56), sharex=True)

# Loop through each source_position and plot the heatmap
for source_position in range(14):
    # Filter the data for the current source_position
    source_data = data[data['source_position'] == source_position]
    
    # Initialize an empty DataFrame to store the combined data for the current source_position
    combined_data = pd.DataFrame()
    
    # Loop through each layer, head pair and concatenate the filtered data
    for layer, head in LayerHead:
        filtered_data = source_data[(source_data['layer'] == layer) & (source_data['head'] == head)]
        filtered_data['layer_head'] = f'Layer {layer} | Head {head}'
        
        # Change the sign for all values with layer 10 and head 7
        if (layer, head) == (10, 7) or (layer, head) == (11, 10):
            filtered_data['value'] = -filtered_data['value']
        
        combined_data = pd.concat([combined_data, filtered_data])
    
    # Set the order of the 'layer_head' category in reverse
    combined_data['layer_head'] = pd.Categorical(combined_data['layer_head'], 
                                                 categories=[f'Layer {layer} | Head {head}' for layer, head in reversed(LayerHead)], 
                                                 ordered=True)
    
    # Pivot the table to get the destination positions as columns
    pivot_table_combined = combined_data.pivot(index='layer_head', columns='dest_position', values='value')
    
    # Plot the heatmap for the combined data
    sns.heatmap(
        pivot_table_combined,
        cmap='seismic',
        center=0,  # Center the colormap at 0
        cbar=False,
        ax=axes[source_position],
        linewidths=0.1,
        linecolor='grey',
        square=True
    )
    
    # Customize the headers
    axes[source_position].set_title(f'Source Position {source_position}', fontsize=12)
    axes[source_position].set_xlabel('')  # Remove the x-axis label for all but the last subplot
    axes[source_position].set_ylabel('')  # Set the y-axis label to empty

# Set the x-axis label for the last subplot
axes[-1].set_xlabel('Destination Position', fontsize=16)

# Adjust the layout
plt.tight_layout()

# Save the figure in the current directory
plt.savefig(os.path.join(current_directory, 'figure combined subplots.png'))

# Show the plot
plt.show()
