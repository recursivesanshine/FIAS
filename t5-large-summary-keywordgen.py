import os
import sys
import glob
import re
import logging
import pandas as pd
import spacy
from transformers import pipeline, AutoTokenizer, T5ForConditionalGeneration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load spaCy English model
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    logger.error("Please install the spaCy model 'en_core_web_sm' using: python -m spacy download en_core_web_sm")
    sys.exit(1)

# Constants
MODEL_NAME = "t5-large"
MAX_INPUT_TOKENS = 512
DEVICE = -1   # Set to 0 if you have a GPU available

# Initialize zero-shot classifier for classifying adjectives.
# Using candidate labels "Atmosphere" and "Emotion".
zero_shot_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=DEVICE
)
# Lower threshold to allow more adjectives through.
CLASSIFICATION_THRESHOLD = 0.5

def extract_keywords(text: str) -> dict:
    """
    Process the input text using spaCy and extract candidate keywords.

    1. For adjectives:
       - We collect each adjective along with its head (the noun it modifies).  
       - If the head of an adjective is one of {"expression", "face", "mood"}, we directly assign it to "Emotion".  
       - Otherwise, we classify the adjective via zero-shot into "Atmosphere" vs. "Emotion".
    2. For Elements:
       - We extract noun chunks that are 2–3 words long.
    
    Return a dictionary with keys "Atmosphere", "Emotion", and "Elements".
    """
    doc = nlp(text)
    
    atmosphere_candidates = set()
    emotion_candidates = set()
    
    # Process adjectives with a dependency heuristic.
    for token in doc:
        if token.pos_ == "ADJ" and token.is_alpha and not token.is_stop and len(token.lemma_) > 2:
            head_text = token.head.text.lower()
            candidate = token.lemma_.lower()
            # If the adjective modifies words like "expression", "face", or "mood", assign to Emotion.
            if head_text in {"expression", "face", "mood"}:
                emotion_candidates.add(candidate)
            else:
                # Otherwise, use zero-shot classification.
                result = zero_shot_classifier(candidate, ["Atmosphere", "Emotion"])
                if result["scores"][0] >= CLASSIFICATION_THRESHOLD:
                    if result["labels"][0] == "Atmosphere":
                        atmosphere_candidates.add(candidate)
                    elif result["labels"][0] == "Emotion":
                        emotion_candidates.add(candidate)
    
    # For Elements, extract noun chunks that are 2 to 3 words long.
    elements_candidates = set()
    for chunk in doc.noun_chunks:
        words = chunk.text.split()
        if 2 <= len(words) <= 3:
            candidate = re.sub(r'[^\w\s]', '', chunk.text).strip().lower()
            # Basic filtering: ignore overly generic phrases.
            if candidate and len(candidate.split()) <= 3:
                elements_candidates.add(candidate)
    
    return {
        "Atmosphere": sorted(list(atmosphere_candidates)),
        "Emotion": sorted(list(emotion_candidates)),
        "Elements": sorted(list(elements_candidates))
    }

class Analyzer:
    def __init__(self):
        # Initialize summarizer using T5-large.
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
        self.summarizer = pipeline(
            "summarization",
            model=MODEL_NAME,
            tokenizer=MODEL_NAME,
            device=DEVICE
        )
        logger.info(f"{MODEL_NAME} loaded for summarization.")

    def generate_summary(self, text: str) -> str:
        """
        Generate a summary of the input text using T5-large’s summarization pipeline.
        The summary is adjusted to exactly 125 characters (padded or truncated),
        a dot appended if missing, and enclosed in double quotes.
        """
        clean_text = re.sub(r"[^\w\s\-.,;:'\"]", "", text).strip()
        summary_result = self.summarizer(clean_text, max_length=60, min_length=30)[0]['summary_text']
        if len(summary_result) < 125:
            summary = summary_result.ljust(125)
        else:
            summary = summary_result[:125]
        summary = summary.rstrip()
        if not summary.endswith('.'):
            summary += '.'
        return '"' + summary + '"'

    def generate_analysis(self, text: str) -> str:
        """
        Generate the final analysis output:
          1. A 125-character summary of the input text.
          2. Extract candidate keywords and classify them into:
             - Atmosphere and Emotion (from adjectives)
             - Elements (from noun chunks)
        """
        summary = self.generate_summary(text)
        keywords = extract_keywords(text)
        atmosphere_str = ", ".join(keywords.get("Atmosphere", [])) or "None"
        emotion_str = ", ".join(keywords.get("Emotion", [])) or "None"
        elements_str = ", ".join(keywords.get("Elements", [])) or "None"
        
        analysis_output = (
            "Summary:\n" + summary + "\n\n" +
            "**Keywords for Atmosphere:**\n" + atmosphere_str + "\n\n" +
            "**Keywords for Emotion:**\n" + emotion_str + "\n\n" +
            "**Elements of the Picture:**\n" + elements_str
        )
        return analysis_output

def process_csv_file(csv_path: str, analyzer: Analyzer):
    """
    Process a CSV file (with a "Description" column) and generate analysis for each row.
    Write the results to an output text file.
    """
    try:
        df = pd.read_csv(csv_path)
        output_lines = []
        for idx, row in df.iterrows():
            content = str(row.get("Description", "")).strip()
            if not content:
                logger.warning(f"Skipping empty row {idx+1}")
                continue
            logger.info(f"Processing row {idx+1}/{len(df)}")
            analysis = analyzer.generate_analysis(content)
            output_lines.append(
                f"Input {idx+1}:\n{content}\n\nOutput:\n{analysis}\n\n{'-' * 40}\n"
            )
        output_file = f"{os.path.splitext(csv_path)[0]}_analysis.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.writelines(output_lines)
        logger.info(f"Saved {len(output_lines)} analyses to {output_file}")
    except Exception as e:
        logger.error(f"File processing failed: {e}")

def main():
    if len(sys.argv) != 2:
        logger.error("Usage: python analyze_images.py <input_csv_or_directory>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    analyzer = Analyzer()
    
    if os.path.isdir(input_path):
        csv_files = glob.glob(os.path.join(input_path, "*.csv"))
        for csv_file in csv_files:
            process_csv_file(csv_file, analyzer)
    else:
        process_csv_file(input_path, analyzer)

if __name__ == "__main__":
    main()
