import sys
import csv
import xml.etree.ElementTree as ET

def load_and_extract_from_xml(file_path, keyword):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            tree = ET.parse(file)
            root = tree.getroot()
            
            concept_ids = []
            terms = []
            
            namespace = {'ns1': 'http://www.lido-schema.org'}  # Replace with the actual namespace URI

            for subject_concept in root.findall(".//ns1:subjectConcept", namespace):
                for concept_id in subject_concept.findall("ns1:conceptID", namespace):
                    if keyword.lower() in concept_id.text.lower():
                        concept_ids.append(concept_id.text)
                        term_pairs = {}
                        for term in subject_concept.findall("ns1:term", namespace):
                            lang = term.attrib.get('{http://www.w3.org/XML/1998/namespace}lang')
                            term_pairs[lang] = term.text
                        if 'de' in term_pairs and 'en' in term_pairs:
                            terms.append((term_pairs['de'], term_pairs['en']))
            
            return concept_ids, terms
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return None, None

def save_to_csv(keyword, concept_ids, terms):
    csv_file = f"{keyword}.csv"
    with open(csv_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Concept ID", "German Term", "English Term"])
        for concept_id, (german_term, english_term) in zip(concept_ids, terms):
            writer.writerow([concept_id, german_term, english_term])
    print(f"Results saved to {csv_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script_name.py path_to_xml_file keyword")
        sys.exit(1)
    
    file_path = sys.argv[1]
    keyword = sys.argv[2]
    
    concept_ids, terms = load_and_extract_from_xml(file_path, keyword)
    print(f"Concept IDs: {concept_ids}")
    print(f"Terms: {terms}")
    
    save_to_csv(keyword, concept_ids, terms)