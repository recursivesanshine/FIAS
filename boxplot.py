import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os

def parse_data(file_path):
    data = []
    category = None
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith('==') and '==' in line:
                category = line.strip('==').strip()
            elif category:
                try:
                    value, keyword = line.split('\t')
                    value = float(value)
                    if 0.00 <= value <= 1.00:  # Ensure the value is within the range
                        data.append([category, value, keyword])
                except ValueError:
                    continue
    return pd.DataFrame(data, columns=['category', 'percentage', 'keyword'])

def create_plot(df):
    # Create a larger figure for better readability
    plt.figure(figsize=(12, 8))  # Adjust the height and width for more space
    
    # Create horizontal boxplots
    sns.boxplot(x="percentage", y="category", data=df, palette="vlag")

    # Customize grid and ticks for better readability
    plt.xticks([i * 0.10 for i in range(11)], [f"{i * 0.10:.2f}" for i in range(11)])  # Set x-axis ticks and labels
    plt.xlim(0, 1)  # Set x-axis range from 0.00 to 1.00
    plt.grid(True, linestyle='--', alpha=0.7)  # Enable grid with dashed lines
    
    # Increase readability of x-axis labels
    plt.tick_params(axis='x', labelsize=10)  # Adjust font size of x-axis labels
    plt.xlabel("Percentage")
    plt.ylabel("Category")
    sns.despine(trim=True, left=True)

def save_and_log_plot(df, csv_file, log_file):
    output_file = os.path.splitext(csv_file)[0] + '_boxplot.png'
    plt.savefig(output_file, bbox_inches='tight')  # Use bbox_inches='tight' to ensure the title is not cut off
    output_path = os.path.abspath(output_file)
    print(f'Output saved at: {output_path}')
    with open(log_file, 'a') as f:
        f.write(f'{output_path}\n')
    plt.show()

def process_files(csv_files, log_file):
    for csv_file in csv_files:
        df = parse_data(csv_file)
        create_plot(df)
        save_and_log_plot(df, csv_file, log_file)

def main():
    parser = argparse.ArgumentParser(description='Create horizontal boxplots from a CSV file or a folder containing CSV files')
    parser.add_argument('input_path', type=str, help='Path to the CSV file or folder containing CSV files')
    args = parser.parse_args()

    log_file = 'output_log.txt'
    with open(log_file, 'w') as f:
        f.write('Output log:\n')

    if os.path.isfile(args.input_path) and args.input_path.endswith('results.csv'):
        # Single CSV file
        csv_files = [args.input_path]
    elif os.path.isdir(args.input_path):
        # Folder containing CSV files
        csv_files = [os.path.join(args.input_path, f) for f in os.listdir(args.input_path) if f.endswith('results.csv')]
    else:
        print('Please provide a valid CSV file or a folder containing CSV files with the suffix "results.csv".')
        return

    process_files(csv_files, log_file)

if __name__ == "__main__":
    main()
