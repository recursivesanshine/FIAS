import os
import csv
import sys
from sentence_transformers import CrossEncoder

def extract_keywords_from_sections(file_path, sections):
    """
    Scans the given text file to find sections (e.g. "**Keywords for Atmosphere:**")
    and extracts the subsequent lines (one per keyword) until an empty line or a new section header is reached.
    
    Args:
        file_path (str): Path to the text file.
        sections (list): List of section headers to search for.
        
    Returns:
        dict: A dictionary mapping the section headline (as extracted) to a list of keywords.
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
                    # Clean up the headline text by removing asterisks and colons.
                    headline = line.strip().strip('*').strip(':')
                    # Get subsequent lines until an empty line or another section header is encountered.
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
    
    The CSV is expected to have the header:
       Keyword,Concept ID,German Term,English Term
    The function uses the first column for matching the category. It maps:
       - "atmosphere" to "atmosphäre"
       - "elements" to "bildelement"
       
    Args:
        csv_file_path (str): Path to the metadata CSV file.
        keyword_category (str): The category name (e.g. "atmosphere", "emotion", "elements").
        
    Returns:
        list: A list of keyword strings for that category.
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
                # Use the English term (last column)
                corpus.append(row[-1].strip())
    return corpus

def get_category_from_headline(headline):
    """
    Determines the keyword category from the section headline.
    If the headline contains information associated with atmosphere, emotion or picture elements,
    it returns "atmosphere", "emotion", or "elements" respectively.
    
    Args:
        headline (str): The section header text.
        
    Returns:
        str or None: The determined category, or None if not determinable.
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

if len(sys.argv) < 3:
    print("Usage: python cross_encoder_metadata.py path/to/your/text_folder path/to/your/csv_metadata_file")
    sys.exit(1)

text_folder_path = sys.argv[1]
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

with open(output_file, 'w', newline='', encoding='utf-8') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(["Cross Encoded Summary"])
    writer.writerow([])

    # Process each text file in the provided text folder.
    for file_name in os.listdir(text_folder_path):
        if not file_name.endswith('.txt'):
            continue

        text_file_path = os.path.join(text_folder_path, file_name)
        writer.writerow([f"Processing file: {text_file_path}"])
        writer.writerow([f"Using CSV metadata file: {csv_metadata_file_path}"])
        writer.writerow([])

        extracted_data = extract_keywords_from_sections(text_file_path, sections)
        for headline, queries in extracted_data.items():
            # Determine the matching category from the headline.
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

            # Process each extracted query line.
            for query in queries:
                # If the query line contains multiple comma-separated keywords, split it.
                query_tokens = [token.strip() for token in query.split(',') if token.strip() and token.strip().lower() != "none"]
                for q in query_tokens:
                    writer.writerow([f"-- Query: {q} --"])
                    print(f"Processing Query: {q}")
                    sentence_combinations = [[q, sentence] for sentence in corpus]
                    if not sentence_combinations:
                        writer.writerow([f"No sentence combinations for query '{q}'. Skipping this query."])
                        continue

                    scores = model.predict(sentence_combinations)
                    sorted_sentences = sorted(zip(scores, corpus), reverse=True, key=lambda x: x[0])
                    
                    for score, sentence in sorted_sentences:
                        result = f"{score:.2f}\t{sentence}"
                        writer.writerow([result])
                        print(result)
                    writer.writerow([])  # Separate queries
            writer.writerow([])  # Separate sections
        writer.writerow([])  # Separate files

print(f"Results saved to {output_file}")
