import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Excel file
data_folder = "network data"
file_path = os.path.join(data_folder, "community_detection_results.xlsx")
df = pd.read_excel(file_path)

# Convert string representations of lists to actual lists
df['community_sizes'] = df['community_sizes'].apply(eval)
df['community_edges'] = df['community_edges'].apply(eval)
df['community_nodes'] = df['community_nodes'].apply(eval)

# Metrics to compare
metrics = ['average_run_time', 'num_communities', 'modularity', 'nmi']
titles = ['Average Run Time', 'Number of Communities', 'Modularity', 'NMI']
y_labels = ['Time (seconds)', 'Number of Communities', 'Modularity', 'NMI']

# Create a folder to save the plots
output_folder = 'algorithm_comparison_plots'
os.makedirs(output_folder, exist_ok=True)

# Function to plot the data
def plot_metrics(df, metrics, titles, y_labels):
    for metric, title, y_label in zip(metrics, titles, y_labels):
        plt.figure(figsize=(14, 8))
        sns.lineplot(data=df, x='algorithm', y=metric, marker='o')
        plt.xlabel('Algorithms', fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.title(title, fontweight='bold')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Remove grid and edge lines for clarity
        plt.grid(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_linewidth(0.5)
        plt.gca().spines['bottom'].set_linewidth(0.5)

        # Highlight important information
        plt.axhline(0, color='gray', linewidth=0.5)
        plt.axvline(-0.5, color='gray', linewidth=0.5)
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.6f}'.format(x) if metric == 'average_run_time' else '{:.2f}'.format(x)))

        plt.savefig(os.path.join(output_folder, f'plot{y_label}.png'))
        plt.show()


    # Bar Plot for each metric
    for metric, title, y_label in zip(metrics, titles, y_labels):
        plt.figure(figsize=(14, 8))
        sns.barplot(data=df, x='algorithm', y=metric)
        plt.xlabel('Algorithms', fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.title(title, fontweight='bold')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'BarPlot{y_label}.png'))
        plt.show()

    # Box Plot for each metric
    for metric, title, y_label in zip(metrics, titles, y_labels):
        plt.figure(figsize=(14, 8))
        sns.boxplot(data=df, x='algorithm', y=metric)
        plt.ylabel(y_label, fontsize=14)
        plt.title(f'Distribution of {title} for Different Algorithms', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'box_plot{y_label}.png'))
        plt.show()
            # Print peak and average values
        peak_value = df[metric].max()
        min_value = df[metric].min()
        average_value = df[metric].mean()
        print(f'{title} - Peak Value: {peak_value:.6f}' if metric == 'average_run_time' else f'{title} - Peak Value: {peak_value:.2f}')
        print(f'{title} - lowest Value: {min_value:.6f}' if metric == 'average_run_time' else f'{title} - min Value: {min_value:.2f}')
        print(f'{title} - Average Value: {average_value:.6f}' if metric == 'average_run_time' else f'{title} - Average Value: {average_value:.2f}')

    # Matrix of Graphs with Annotations for Peak Values
    num_metrics = len(metrics)
    fig, axs = plt.subplots(num_metrics, 1, figsize=(14, 4 * num_metrics), constrained_layout=True)
    fig.suptitle('Execution Metrics of Different Algorithms', fontsize=16, fontweight='bold')

    for i, (metric, title, y_label) in enumerate(zip(metrics, titles, y_labels)):
        max_value = df[metric].max()
        max_index = df[metric].idxmax()
        sns.lineplot(data=df, x='algorithm', y=metric, marker='o', ax=axs[i])
        axs[i].annotate(f'Peak: {max_value:.6f}' if metric == 'average_run_time' else f'Peak: {max_value:.2f}', 
                        xy=(max_index, max_value), 
                        xytext=(max_index, max_value * 1.1 if max_value * 1.1 > max_value else max_value * 0.9),
                        arrowprops=dict(facecolor='black', shrink=0.05),
                        fontsize=10, color='red', fontweight='bold')
        axs[i].set_xlabel('Algorithms', fontsize=12)
        axs[i].set_ylabel(y_label, fontsize=12)
        axs[i].set_title(title, fontsize=14)
        axs[i].grid(False)
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['left'].set_linewidth(0.5)
        axs[i].spines['bottom'].set_linewidth(0.5)
        axs[i].axhline(0, color='gray', linewidth=0.5)
        axs[i].axvline(-0.5, color='gray', linewidth=0.5)
        axs[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.6f}'.format(x) if metric == 'average_run_time' else '{:.2f}'.format(x)))
        axs[i].tick_params(axis='x', rotation=45)

    plt.savefig(os.path.join(output_folder, f'matrix_of_graphs{y_label}.png'))
    plt.show()

# Plot the data
plot_metrics(df, metrics, titles, y_labels)
