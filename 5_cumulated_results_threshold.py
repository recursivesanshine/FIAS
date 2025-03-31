import sys
import os

def is_score_line(line):
    """
    Checks if a line starts with a numeric value.
    Returns a tuple (True, score) if it does, or (False, None) otherwise.
    """
    stripped = line.strip()
    if not stripped:
        return False, None
    # Split on whitespace (tabs or spaces)
    tokens = stripped.split()
    try:
        score = float(tokens[0])
        return True, score
    except ValueError:
        return False, None

def main():
    # Check if user provided an input file.
    if len(sys.argv) < 2:
        print("Usage: python cumulated_results_threshold.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    # Derive an output file name by appending _threshold.
    base_name, ext = os.path.splitext(input_file)
    output_file = f"{base_name}_threshold{ext}"
    
    # Threshold value
    threshold = 0.50

    try:
        with open(input_file, "r", encoding="utf-8") as infile, \
             open(output_file, "w", encoding="utf-8") as outfile:
            
            for line in infile:
                # Check if the line begins with a numeric token.
                numeric_line, score = is_score_line(line)
                if numeric_line:
                    # Write out the line only if the score is >= threshold.
                    if score >= threshold:
                        outfile.write(line)
                    else:
                        # Skip this line.
                        continue
                else:
                    # This line isn't a numeric data row, so we write it as is.
                    outfile.write(line)
        
        # Print output file details.
        abs_output_path = os.path.abspath(output_file)
        print(f"Filtered output written to: '{output_file}'")
        print(f"Full path: {abs_output_path}")
    
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
