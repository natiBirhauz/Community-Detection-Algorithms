import os
import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel file
data_folder = "network data"
file_path = os.path.join(data_folder, "testing.xlsx")
df = pd.read_excel(file_path)

# Set the first column as the index (network types)
df.set_index(df.columns[0], inplace=True)
# Exclude the "JN" column (Girvan-Newman algorithm)
df = df.drop(columns=['JN'])
# Plotting the data - Line Plot
plt.figure(figsize=(14, 8))
for column in df.columns:
    plt.plot(df.index, df[column], marker='o', label=column)

plt.xlabel('Network Types', fontsize=14)
plt.ylabel('Time (seconds)', fontsize=14)
plt.title('Time to Execution Algorithms on Different Network Types', fontweight='bold')
plt.xticks(rotation=45)
plt.legend(title='Algorithms', loc='lower left', fontsize='x-small')
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
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.2f}'.format(x)))

plt.show()

# Bar Plot
plt.figure(figsize=(14, 8))
df.plot(kind='bar', figsize=(14, 8))
plt.xlabel('Network Types', fontsize=14)
plt.ylabel('Time (seconds)', fontsize=14)
plt.title('Time to Execution Algorithms on Different Network Types', fontweight='bold')
plt.xticks(rotation=45)
plt.legend(title='Algorithms', fontsize='x-small')
plt.tight_layout()
plt.show()

# Box Plot
plt.figure(figsize=(14, 8))
df.plot(kind='box', figsize=(14, 8))
plt.ylabel('Time (seconds)', fontsize=14)
plt.title('Distribution of Execution Times for Different Algorithms', fontweight='bold')
plt.tight_layout()
plt.show()

# Matrix of Graphs
num_columns = len(df.columns)
fig, axs = plt.subplots(num_columns, 1, figsize=(14, 4 * num_columns), constrained_layout=True)
fig.suptitle('Execution Time of Different Algorithms', fontsize=16, fontweight='bold')

for i, column in enumerate(df.columns):
     # Annotate the peak time
    max_value = df[column].max()
    max_index = df[column].idxmax()
    axs[i].annotate(f'Peak: {max_value:.2f}', xy=(max_index, max_value), xytext=(max_index, max_value*1.1),
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    fontsize=10, color='red', fontweight='bold')
    
    axs[i].plot(df.index, df[column], marker='o', label=column)
    if i == num_columns - 1:
        axs[i].set_xlabel('Network Types', fontsize=12)
        axs[i].set_ylabel('Time (seconds)', fontsize=12)

    else:
        axs[i].set_xticklabels([])
    axs[i].set_title(column, fontsize=14)
    axs[i].legend(loc='upper right')
    axs[i].grid(False)
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    axs[i].spines['left'].set_linewidth(0.5)
    axs[i].spines['bottom'].set_linewidth(0.5)
    axs[i].axhline(0, color='gray', linewidth=0.5)
    axs[i].axvline(-0.5, color='gray', linewidth=0.5)
    axs[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.2f}'.format(x)))
    axs[i].tick_params(axis='x', rotation=45)

plt.show()
