from transformers import pipeline, AutoTokenizer, T5ForConditionalGeneration
import pandas as pd
import os
import sys
import glob
import re
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Constants
MODEL_NAME = 't5-large'
MAX_INPUT_TOKENS = 512
DEVICE = -1  # Set to 0 for GPU usage

PROMPT_TEMPLATE = (
    "Generate output in EXACTLY this format:\n"
    "The image features... [3-sentence description]\n\n"
    "**Keywords for Atmosphere:**\n- Historical\n- Formal\n- Elegant\n- Somber\n- Traditional\n\n"
    "**Keywords for Emotion:**\n- Serene\n- Noble\n- Calm\n- Reserved\n- Dignified\n\n"
    "**Elements of the Picture:**\n- Oil painting\n- Lace collar\n- Pearl jewelry\n- Wooden chair\n- Dark background\n- Intricate shading\n- Textured fabric\n\n"
    "STRICT RULES:\n"
    "1. Use ONLY terms from text\n"
    "2. No explanations/instructions\n"
    "3. Capitalize first letters\n"
    "4. Maintain section order\n\n"
    "Input text: {content}"
)

class SummaryGenerator:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
        self.generator = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=DEVICE
        )
        logger.info(f"{MODEL_NAME} model loaded")

    def _calculate_max_content_tokens(self):
        """Calculate available tokens for content after accounting for prompt"""
        prompt_only = PROMPT_TEMPLATE.format(content="")
        inputs = self.tokenizer(prompt_only, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS)
        return MAX_INPUT_TOKENS - inputs.input_ids.shape[1]

    def _clean_keyword(self, keyword: str) -> str:
        """Validate and clean individual keywords with enhanced rules"""
        cleaned = re.sub(r'[^a-zA-Z\- ]', '', keyword).strip()
        cleaned = ' '.join([word.capitalize() for word in cleaned.split()])
        if 2 < len(cleaned) < 50 and cleaned[0].isupper():
            return f"- {cleaned}"
        return ""

    def _parse_section(self, text: str, header: str) -> list:
        """Robust section parsing with multiple fallback patterns"""
        patterns = []
        
        # Pattern 1: Standard header format
        patterns.append(rf"\*\*{re.escape(header)}:\*\*[\s\n]*(.*?)(?=\n\*\*|\n\n|$)")
        
        # Pattern 2: Handle whitespace variations with raw string
        modified_header = re.escape(header.replace(' ', r'\s+'))
        patterns.append(rf"(?i){modified_header}:?[\s\n]*(.*?)(?=\n\*\*|\n\n|$)")
        
        # Pattern 3: Alternative header format
        patterns.append(rf"{re.escape(header)}\s+Keywords:[\s\n]*(.*?)(?=\n\*\*|\n\n|$)")

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                keywords = []
                for line in match.group(1).split('\n'):
                    line = re.sub(r'^\s*-\s*', '', line.strip())
                    if line:
                        keywords.extend([self._clean_keyword(kw) for kw in re.split(r',|\s*-\s*', line)])
                return list(filter(None, keywords))[:7]
        return []

    def _generate_fallback_keywords(self, description: str, category: str) -> list:
        """Enhanced fallback keyword generation"""
        prompt = (
            f"Extract 7 {category} terms from this art description. "
            f"Focus on concrete elements for 'Elements', mood for 'Atmosphere', "
            f"and human emotions for 'Emotion'.\n\nDescription: {description[:300]}\n"
            "Format: Comma-separated list, no numbers"
        )
        try:
            output = self.generator(
                prompt,
                max_length=100,
                num_beams=2,
                early_stopping=True,
                no_repeat_ngram_size=2
            )[0]['generated_text']
            return [self._clean_keyword(kw) for kw in output.split(',')[:7] if kw.strip()]
        except Exception as e:
            logger.error(f"Fallback generation failed: {e}")
            return []

    def generate_analysis(self, text: str) -> str:
        """Generate complete analysis with enhanced validation"""
        try:
            max_content_tokens = self._calculate_max_content_tokens()
            
            inputs = self.tokenizer(
                text,
                truncation=True,
                max_length=max_content_tokens,
                return_tensors="pt"
            )
            truncated_content = self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
            
            response = self.generator(
                PROMPT_TEMPLATE.format(content=truncated_content),
                max_length=512,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
                repetition_penalty=1.5
            )[0]['generated_text']

            desc_match = re.search(r'^(The image features.*?)(?=\n\*\*|\n\n)', response, re.DOTALL)
            description = desc_match.group(1).strip() if desc_match else text[:300] + "..."

            sections = {
                'atmosphere': self._parse_section(response, "Keywords for Atmosphere") or 
                             self._generate_fallback_keywords(description, "atmospheric"),
                'emotion': self._parse_section(response, "Keywords for Emotion") or 
                          self._generate_fallback_keywords(description, "emotional"),
                'elements': self._parse_section(response, "Elements of the Picture") or 
                            self._generate_fallback_keywords(description, "visual elements")
            }

            for sect in ['atmosphere', 'emotion']:
                sections[sect] = sections[sect][:5] or self._generate_fallback_keywords(description, sect)
            sections['elements'] = sections['elements'][:7] or self._generate_fallback_keywords(description, "elements")

            return (
                f"{description}\n\n"
                f"**Keywords for Atmosphere:**\n" + '\n'.join(sections['atmosphere'][:5]) + "\n\n"
                f"**Keywords for Emotion:**\n" + '\n'.join(sections['emotion'][:5]) + "\n\n"
                f"**Elements of the Picture:**\n" + '\n'.join(sections['elements'][:7])
            )

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return "Analysis error - invalid output format"

def process_csv_file(csv_path: str, generator: SummaryGenerator):
    """Process CSV file with enhanced validation and error handling"""
    try:
        df = pd.read_csv(csv_path)
        output_lines = []
        
        for idx, row in df.iterrows():
            content = str(row['Description']).strip()
            if not content:
                logger.warning(f"Skipping empty row {idx+1}")
                continue

            logger.info(f"Processing row {idx+1}/{len(df)}")
            analysis = generator.generate_analysis(content)
            output_lines.append(f"Input {idx+1}:\n{content}\n\nAnalysis:\n{analysis}\n\n{'-'*40}\n")

        output_file = f"{os.path.splitext(csv_path)[0]}_analysis.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(output_lines)
            
        logger.info(f"Saved {len(output_lines)} analyses to {output_file}")

    except Exception as e:
        logger.error(f"File processing failed: {str(e)}")

def main():
    if len(sys.argv) != 2:
        logger.error("Usage: python analyze_images.py <input_csv_or_directory>")
        sys.exit(1)

    input_path = sys.argv[1]
    generator = SummaryGenerator()

    if os.path.isdir(input_path):
        csv_files = glob.glob(os.path.join(input_path, '*.csv'))
        for csv_file in csv_files:
            process_csv_file(csv_file, generator)
    else:
        process_csv_file(input_path, generator)

if __name__ == "__main__":
    main()