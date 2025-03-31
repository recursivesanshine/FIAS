import os
import sys
import re
import pandas as pd

def build_matches(folder_path):
    """
    Scans every CSV file in folder_path for matching keywords.
    Each CSV is expected to have an "English Term" column and optionally a "Percentage" column.
    
    Returns a dictionary mapping lower-case keyword -> list of (filename, percentage) tuples.
    """
    matches = {}
    
    # Loop over all CSV files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            try:
                df = pd.read_csv(file_path, on_bad_lines='skip')
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
            for _, row in df.iterrows():
                eng = row.get("English Term", "")
                if not isinstance(eng, str):
                    continue
                eng_lower = eng.strip().lower()
                percentage = row.get("Percentage", "")
                # Convert percentage to string (if not already) so we can print it
                if pd.notnull(percentage):
                    percentage = str(percentage)
                else:
                    percentage = ""
                if eng_lower not in matches:
                    matches[eng_lower] = []
                matches[eng_lower].append((filename, percentage))
    return matches

def update_threshold_file(threshold_file, matches, output_file):
    """
    Reads the threshold file (keeping its sections and structure) and for every line
    that starts with a percentage followed by a keyword (e.g. "0.58 heaviness"),
    appends a list of matching CSV occurrences from the matches dictionary.
    
    Lines that do not match that pattern are written unchanged.
    """
    # This regex matches a line that starts (optionally with spaces) with a floating-point number,
    # followed by whitespace and then some text. (This identifies lines like "0.58  heaviness")
    pattern = re.compile(r'^(\s*\d+\.\d+)\s+(.*)$')
    output_lines = []
    
    with open(threshold_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            m = pattern.match(line)
            if m:
                # m.group(1) contains the percentage part; m.group(2) contains the keyword text.
                perc_text = m.group(1)
                keyword_text = m.group(2).strip()
                keyword_lower = keyword_text.lower()
                if keyword_lower in matches:
                    found_entries = matches[keyword_lower]
                    # Create a string list like "file1.csv (0.58)", "file2.csv (0.62)"
                    entries_str = ", ".join([f"{fn} ({pt})" for fn, pt in found_entries if pt])
                    # If percentage values are missing (empty string), you might also list just filenames:
                    if not entries_str:
                        entries_str = ", ".join([fn for fn, pt in found_entries])
                    new_line = f"{perc_text}\t{keyword_text} [Found: {entries_str}]"
                    output_lines.append(new_line)
                else:
                    # If no match is found, write the line as-is.
                    output_lines.append(line)
            else:
                # Leave non-matching lines (like section headers, query lines, etc.) unchanged.
                output_lines.append(line)
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for outline in output_lines:
            f_out.write(outline + "\n")
    
    print(f"Updated file saved as {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python 8_matching_last_keywords.py <threshold_file> <folder_path>")
        sys.exit(1)
    
    threshold_file = sys.argv[1]
    folder_path = sys.argv[2]
    output_file = "updated_cross_encoded_summary.txt"
    
    print("Scanning CSV files for matching keywords...")
    matches = build_matches(folder_path)
    # Debug: show the built matches dictionary (you can comment this out if too verbose)
    print("Matches found:")
    for key, val in matches.items():
        print(f"{key}: {val}")
    
    print("Updating the summary file with CSV matches...")
    update_threshold_file(threshold_file, matches, output_file)
