import argparse
import base64
import io
import json
import os
import pandas as pd
from PIL import Image

def image_to_base64(image_data):
    """
    Converts image data (bytes or dict with bytes) to a base64 encoded JPEG string.
    """
    try:
        if isinstance(image_data, dict):
            if 'bytes' in image_data and image_data['bytes'] is not None:
                img_bytes = image_data['bytes']
            elif 'path' in image_data and os.path.exists(image_data['path']):
                with open(image_data['path'], 'rb') as f:
                    img_bytes = f.read()
            else:
                return None
        elif isinstance(image_data, bytes):
            img_bytes = image_data
        else:
            return None
            
        # Open with PIL to ensure it's a valid image and convert to JPEG
        img = Image.open(io.BytesIO(img_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Transform a Parquet file into a JSONL file for Together AI VLM fine-tuning.")
    parser.add_argument("input_parquet", help="Path to the input Parquet file.")
    parser.add_argument("--output_jsonl", help="Path to the output JSONL file. Defaults to input path with .jsonl extension.")
    parser.add_argument("--prompt", default="Classify this worksheet.", help="The user prompt to use for each example.")
    parser.add_argument("--prompt_file", help="Optional path to a file containing the full user prompt.")
    
    args = parser.parse_args()
    
    # Determine the prompt
    prompt_text = args.prompt
    if args.prompt_file:
        if os.path.exists(args.prompt_file):
            with open(args.prompt_file, 'r', encoding='utf-8') as f:
                prompt_text = f.read().strip()
        else:
            print(f"Warning: Prompt file {args.prompt_file} not found. Using default prompt.")
            
    # Determine the output path
    output_path = args.output_jsonl
    if not output_path:
        output_path = os.path.splitext(args.input_parquet)[0] + ".jsonl"
        
    print(f"Reading Parquet file: {args.input_parquet}")
    try:
        df = pd.read_parquet(args.input_parquet)
    except Exception as e:
        print(f"Error reading Parquet file: {e}")
        return

    # Identify columns
    # Common names: 'image', 'labels', 'label', 'text', 'markdown'
    image_col = next((c for c in df.columns if 'image' in c.lower()), None)
    label_col = next((c for c in df.columns if c.lower() in ['labels', 'label', 'response', 'answer']), None)
    
    if not image_col:
        print(f"Available columns: {df.columns.tolist()}")
        print("Error: Could not find an image column.")
        return
        
    if not label_col:
        print(f"Warning: Could not find a label column. Using empty strings for assistant responses.")

    print(f"Using image column: '{image_col}'")
    if label_col:
        print(f"Using label column: '{label_col}'")
    
    count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, row in df.iterrows():
            b64_image = image_to_base64(row[image_col])
            if not b64_image:
                print(f"Skipping row {i} due to missing or invalid image data.")
                continue
                
            assistant_text = str(row[label_col]) if label_col and row[label_col] is not None else ""
            
            # Construct Together AI VLM instruction format
            entry = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64_image}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt_text
                            }
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": assistant_text
                            }
                        ]
                    }
                ]
            }
            f.write(json.dumps(entry) + '\n')
            count += 1
            if count % 100 == 0:
                print(f"Processed {count} rows...")
                
    print(f"Successfully converted {count} rows to {output_path}")

if __name__ == "__main__":
    main()
