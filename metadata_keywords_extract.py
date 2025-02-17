import sys
import csv
import xml.etree.ElementTree as ET
import logging
import os
from typing import List, Tuple, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the keywords to search for (case-insensitive)
KEYWORDS = ["motiv", "bildelement", "assoziation", "atmosphäre", "emotion"]

def load_and_extract_from_xml(file_path: str) -> Dict[str, List[Tuple[str, str, str]]]:
    """
    Load and extract data from an XML file for all keywords.

    Args:
        file_path (str): Path to the XML file.

    Returns:
        Dict[str, List[Tuple[str, str, str]]]: A dictionary where keys are keywords and values are lists of tuples (Concept ID, German Term, English Term).
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            tree = ET.parse(file)
            root = tree.getroot()
            
            data_by_keyword = {keyword: [] for keyword in KEYWORDS}
            
            namespace = {'ns1': 'http://www.lido-schema.org'}  # Replace with the actual namespace URI
            logging.info(f"Using namespace: {namespace}")

            for subject_concept in root.findall(".//ns1:subjectConcept", namespace):
                for concept_id in subject_concept.findall("ns1:conceptID", namespace):
                    concept_id_text = concept_id.text.lower()
                    for keyword in KEYWORDS:
                        if keyword.lower() in concept_id_text:
                            logging.info(f"Found keyword '{keyword}' in conceptID: {concept_id_text}")
                            term_pairs = {}
                            for term in subject_concept.findall("ns1:term", namespace):
                                lang = term.attrib.get('{http://www.w3.org/XML/1998/namespace}lang')
                                term_pairs[lang] = term.text
                                logging.info(f"Extracted term: {term.text} (lang: {lang})")
                            if 'de' in term_pairs and 'en' in term_pairs:
                                data_by_keyword[keyword].append((concept_id.text, term_pairs['de'], term_pairs['en']))
            
            return data_by_keyword
    except ET.ParseError as e:
        logging.error(f"Error parsing XML file: {e}")
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    
    return {}

def save_to_csv(output_file_path: str, data_by_keyword: Dict[str, List[Tuple[str, str, str]]]) -> None:
    """
    Save extracted data to a single CSV file, categorized by keyword.

    Args:
        output_file_path (str): Path to the output CSV file.
        data_by_keyword (Dict[str, List[Tuple[str, str, str]]]): Data organized by keyword.
    """
    try:
        if not any(data_by_keyword.values()):  # Skip if there are no matching results
            logging.info(f"No matching results found. Skipping file: {output_file_path}")
            return

        with open(output_file_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Keyword", "Concept ID", "German Term", "English Term"])

            for keyword, rows in data_by_keyword.items():
                for row in rows:
                    writer.writerow([keyword] + list(row))

        logging.info(f"Results saved to {output_file_path}")
    except Exception as e:
        logging.error(f"Error saving to CSV file: {e}")

def process_xml_file(file_path: str) -> None:
    """
    Process a single XML file to extract data and save to a CSV file.

    Args:
        file_path (str): Path to the XML file.
    """
    # Derive the base name of the XML file (without extension)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Define the output CSV file path using the base name
    output_csv_path = f"{os.path.join(os.path.dirname(file_path), base_name)}.csv"
    
    # Load and extract data from the XML file
    data_by_keyword = load_and_extract_from_xml(file_path)
    
    if not any(data_by_keyword.values()):  # Skip if there are no matching results
        logging.info(f"No matching results found in {file_path}. Skipping.")
        return
    
    # Log processing details for each keyword
    for keyword in KEYWORDS:
        logging.info(f"Processing keyword: {keyword} in file: {os.path.basename(file_path)}")
        concept_ids = [row[0] for row in data_by_keyword[keyword]]
        terms = [(row[1], row[2]) for row in data_by_keyword[keyword]]
        
        if concept_ids and terms:
            logging.info(f"Concept IDs for '{keyword}': {concept_ids}")
            logging.info(f"Terms for '{keyword}': {terms}")
        else:
            logging.warning(f"No data found for keyword: {keyword} in file: {os.path.basename(file_path)}")
    
    # Save combined data into a CSV file named after the XML file
    save_to_csv(output_csv_path, data_by_keyword)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        logging.error("Usage: metadata_keyword_extract.py path_to_xml_file_or_folder")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    if os.path.isdir(input_path):
        # Process all XML files in the specified folder
        for file_name in os.listdir(input_path):
            if file_name.endswith('.xml'):
                file_path = os.path.join(input_path, file_name)
                process_xml_file(file_path)
    elif os.path.isfile(input_path) and input_path.endswith('.xml'):
        # Process a single XML file
        process_xml_file(input_path)
    else:
        logging.error("Invalid input path. Please provide a path to a directory or an XML file.")