import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os

def parse_data(file_path):
    """
    Parse the input file, capturing headers and data lines.
    Returns a DataFrame with columns:
    - main_category (e.g., Atmosphere, Emotion, Picture Elements, Association, or Motive)
    - subcategory (the complete section header)
    - percentage (the numeric value)
    - keyword (the associated word)
    """
    data = []
    category = None
    main_category = None
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            # Detect section headers like "== Keywords for Atmosphere ==" or similar variants.
            if line.startswith("==") and line.endswith("=="):
                category = line.strip("=").strip()
                if 'Atmosphere' in category:
                    main_category = 'Atmosphere'
                elif 'Emotion' in category:
                    main_category = 'Emotion'
                elif 'Association' in category:
                    main_category = 'Association'
                elif 'Motive' in category:
                    main_category = 'Motive'
                # Handle variations for picture elements.
                elif 'Picture Elements' in category or 'Elements of the Picture' in category:
                    main_category = 'Picture Elements'
                else:
                    main_category = None  # Skip if section is not one of the expected ones.
            # Process data lines only when a main_category is set and line contains a tab.
            elif main_category and "\t" in line:
                try:
                    value, keyword = line.split('\t')
                    value = float(value)
                    if 0.00 <= value <= 1.00:
                        data.append([main_category, category, value, keyword])
                except ValueError:
                    continue
    return pd.DataFrame(data, columns=['main_category', 'subcategory', 'percentage', 'keyword'])

def visualize_distribution(df, category):
    """
    Create a boxplot for subcategories within a specified main category.
    """
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="percentage", y="subcategory", data=df[df['main_category'] == category], palette="vlag")
    plt.title(f'Distribution of Subcategories within {category}')
    plt.show()

def create_plot(df):
    """
    Create a boxplot showing the overall distribution of percentage values across all main categories.
    The plot will include all five categories: Atmosphere, Emotion, Picture Elements, Association, and Motive.
    """
    desired_categories = ["Atmosphere", "Emotion", "Picture Elements", "Association", "Motive"]
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="percentage", y="main_category", data=df, order=desired_categories, palette="vlag")
    plt.xticks([i * 0.10 for i in range(11)], [f"{i * 0.10:.2f}" for i in range(11)])
    plt.xlim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tick_params(axis='x', labelsize=10)
    plt.xlabel("Percentage")
    plt.ylabel("Category")
    sns.despine(trim=True, left=True)

def save_and_log_plot(df, log_file):
    """
    Save the current plot as a PNG file and log its full path in a log file.
    """
    output_file = 'cumulated_results_boxplot.png'
    plt.savefig(output_file, bbox_inches='tight')
    output_path = os.path.abspath(output_file)
    print(f'Output saved at: {output_path}')
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f'{output_path}\n')
    plt.show()

def process_files(csv_files, log_file):
    """
    Process all CSV files provided:
    - Parse each file and combine their data.
    - Create and save the overall boxplot showing all five categories.
    - Additionally, visualize the distribution for "Picture Elements".
    """
    combined_df = pd.DataFrame(columns=['main_category', 'subcategory', 'percentage', 'keyword'])
    for csv_file in csv_files:
        df = parse_data(csv_file)
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    create_plot(combined_df)
    save_and_log_plot(combined_df, log_file)
    visualize_distribution(combined_df, "Picture Elements")

def main():
    parser = argparse.ArgumentParser(
        description='Create boxplots from CSV files containing cross encoded summary data'
    )
    parser.add_argument('input_path', type=str, help='Path to a CSV file or to a folder containing CSV files')
    args = parser.parse_args()

    log_file = 'output_log.txt'
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write('Output log:\n')

    csv_files = []
    # Check if the provided input is a CSV file or a folder containing CSV files.
    if os.path.isfile(args.input_path) and args.input_path.lower().endswith(".csv"):
        csv_files = [args.input_path]
    elif os.path.isdir(args.input_path):
        csv_files = [
            os.path.join(args.input_path, f)
            for f in os.listdir(args.input_path)
            if f.lower().endswith(".csv")
        ]
    else:
        print('Please provide a valid CSV file or a folder containing CSV files.')
        return

    process_files(csv_files, log_file)

if __name__ == "__main__":
    main()
