import os
import re
from collections import defaultdict

def extract_current_csv(processing_line):
    """Extracts the CSV filename from a 'Processing file:' line."""
    parts = processing_line.split("Processing file:")
    if len(parts) < 2:
        return None
    file_str = parts[1].strip()
    base = os.path.basename(file_str)
    if base.endswith("_output_analysis.txt"):
        return base.replace("_output_analysis.txt", ".csv")
    return base

def process_summary_file(input_file, output_file):
    """Processes the summary file with all requested filters and grouping."""
    # Make sure the regex pattern is entirely on one line.
    pattern = re.compile(r'^(?P<perc>\s*\d+\.\d+)\s+(?P<keyword>.*?)\s+\[Found:\s*(?P<found>.*?)\]$')
    
    # Dictionary structure: {csv: {category: {keyword: max_perc}}}
    results = defaultdict(lambda: defaultdict(dict))
    current_csv = None
    current_category = None

    with open(input_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.rstrip('\n')
            
            # Update current CSV from processing file lines.
            if line.startswith("Processing file:"):
                current_csv = extract_current_csv(line)
                current_category = None
                continue
            
            # Set the current category based on section headers.
            if line.startswith("== "):
                current_category = line.strip("= ").strip()
                continue
            
            # Skip query lines.
            if line.startswith("-- Query:"):
                continue
            
            m = pattern.match(line)
            if m and current_csv and current_category:
                keyword = m.group("keyword").strip()
                found_text = m.group("found").strip()
                perc = float(m.group("perc"))
                
                # Process only matches originating from the same file.
                if any(current_csv == item.strip() for item in found_text.split(",")):
                    # Keep only the highest percentage for each keyword.
                    if (keyword not in results[current_csv][current_category] or 
                        perc > results[current_csv][current_category][keyword]):
                        results[current_csv][current_category][keyword] = perc
    
    # Generate the report.
    output_lines = []
    for csv, categories in results.items():
        output_lines.append(f"\nProcessing file: {csv}")
        output_lines.append("=" * (len(f"Processing file: {csv}") + 1))
        
        total_matches = sum(len(keywords) for keywords in categories.values())
        output_lines.append(f"\nTotal unique matches: {total_matches}")
        
        # Process categories in the specified order (all 5 categories).
        category_order = [
            "Keywords for Atmosphere",
            "Keywords for Emotion", 
            "Elements of the Picture",
            "Keywords for Association",
            "Keywords for Motive"
        ]
        
        for category in category_order:
            if category in categories:
                # Sort keywords by percentage (highest first).
                sorted_keywords = sorted(
                    categories[category].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                if sorted_keywords:
                    output_lines.append(f"\n== {category} ==")
                    for keyword, perc in sorted_keywords:
                        output_lines.append(f"{perc:.2f}\t{keyword}")
        
        output_lines.append("")
    
    with open(output_file, 'w', encoding='utf-8') as fout:
        fout.write("\n".join(output_lines))
    
    print(f"Final filtered and grouped summary saved to {output_file}")

if __name__ == "__main__":
    INPUT_FILE = "updated_cross_encoded_summary.txt"
    OUTPUT_FILE = "final_grouped_results.txt"
    process_summary_file(INPUT_FILE, OUTPUT_FILE)
