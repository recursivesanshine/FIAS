import os
import csv
import sys
from sentence_transformers import CrossEncoder

def extract_keywords_from_sections(file_path, sections):
    """
    Scans the given text file to find sections (e.g. "**Keywords for Atmosphere:**")
    and extracts the subsequent lines (one per keyword) until an empty line or a new section header is reached.
    """
    extracted_data = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for section in sections:
            keywords = []
            section_found = False
            for i, line in enumerate(lines):
                if f"**{section}:**" in line:
                    section_found = True
                    headline = line.strip().strip('*').strip(':')
                    for keyword_line in lines[i+1:]:
                        if keyword_line.strip() == "" or keyword_line.strip().startswith('**'):
                            break
                        keywords.append(keyword_line.strip().strip('-').strip())
                    extracted_data[headline] = keywords
                    break
            if not section_found:
                print(f"Section '{section}' not found in {file_path}.")
    return extracted_data

def extract_corpus_from_csv(csv_file_path, keyword_category):
    """
    Given the CSV metadata file and a keyword category, returns the list of English keywords.
    """
    corpus = []
    keyword_category = keyword_category.lower()
    if keyword_category == "atmosphere":
        keyword_category = "atmosphäre"
    elif keyword_category == "elements":
        keyword_category = "bildelement"
    
    with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Skip the header row
        for row in reader:
            if row[0].strip().lower() == keyword_category:
                corpus.append(row[-1].strip())
    return corpus

def get_category_from_headline(headline):
    """
    Determines the keyword category from the section headline.
    """
    low = headline.lower()
    if "atmosphere" in low or "atmosphäre" in low:
        return "atmosphere"
    elif "emotion" in low:
        return "emotion"
    elif "element" in low or "picture" in low:
        return "elements"
    else:
        return None

def main():
    if len(sys.argv) < 3:
        print("Usage: python cross_encoder_metadata.py path/to/your/text_folder_or_file path/to/your/csv_metadata_file")
        sys.exit(1)

    input_path = sys.argv[1]
    csv_metadata_file_path = sys.argv[2]
    output_file = "cross_encoded_summary.csv"

    # Initialize the cross encoder model.
    model = CrossEncoder("cross-encoder/stsb-distilroberta-base")

    # Define potential section header variants.
    sections = [
        "Keywords for Atmosphere", 
        "Keywords for Emotion", 
        "Picture Elements", 
        "Elements of the Picture"
    ]
    
    # Open the output file with tab as the delimiter.
    with open(output_file, 'w', newline='', encoding='utf-8') as outcsv:
        writer = csv.writer(outcsv, delimiter='\t', quoting=csv.QUOTE_NONE)
        writer.writerow(["Cross Encoded Summary"])
        writer.writerow([])

        # Check if the input path is a file or a directory.
        if os.path.isfile(input_path):
            file_list = [input_path]  # Single file
        elif os.path.isdir(input_path):
            file_list = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.txt')]  # Directory
        else:
            print(f"Invalid input path: {input_path}")
            sys.exit(1)

        # Process each file.
        for file_path in file_list:
            writer.writerow([f"Processing file: {file_path}"])
            writer.writerow([f"Using CSV metadata file: {csv_metadata_file_path}"])
            writer.writerow([])

            extracted_data = extract_keywords_from_sections(file_path, sections)
            for headline, queries in extracted_data.items():
                category = get_category_from_headline(headline)
                if not category:
                    writer.writerow([f"Could not determine category from headline: {headline}. Skipping."])
                    writer.writerow([])
                    continue

                corpus = extract_corpus_from_csv(csv_metadata_file_path, category)
                writer.writerow([f"== {headline} =="])
                print(f"Corpus for category '{category}': {corpus}")

                if not corpus:
                    writer.writerow([f"No corpus found for keyword category '{category}'. Skipping this section."])
                    writer.writerow([])
                    continue

                for query in queries:
                    query_tokens = [token.strip() for token in query.split(',') if token.strip() and token.strip().lower() != "none"]
                    for q in query_tokens:
                        writer.writerow([f"-- Query: {q} --"])
                        print(f"Processing Query: {q}")
                        sentence_combinations = [[q, sentence] for sentence in corpus]
                        if not sentence_combinations:
                            writer.writerow([f"No sentence combinations for query '{q}'. Skipping this query."])
                            continue

                        scores = model.predict(sentence_combinations)
                        sorted_sentences = sorted(
                            zip(scores, corpus),
                            reverse=True,
                            key=lambda x: x[0]
                        )

                        for score, sentence in sorted_sentences:
                            writer.writerow([f"{score:.2f}", sentence])
                            print(f"{score:.2f}\t{sentence}")
                        writer.writerow([])  # Separate queries
                writer.writerow([])  # Separate sections
            writer.writerow([])  # Separate files

    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
