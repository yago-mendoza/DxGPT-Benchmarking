#!/usr/bin/env python3
"""
ETL Script for formatting diagnosis names in ramedis_bench dataset.

This script processes the 'diagnosis' column to extract and format multiple disease names:
- Splits diagnoses by " also known as " or " or " 
- Creates a semicolon-separated list of disease names
- Makes the diagnosis column more concise and readable

Input: before.csv
Output: after.csv
"""

import pandas as pd
import re
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Read the input CSV from the same directory
df = pd.read_csv(os.path.join(script_dir, 'before.csv'))

def format_diagnosis(diagnosis_text):
    """
    Format diagnosis text by extracting and cleaning disease names.
    
    Input: "Disease A also known as Disease B or Disease C"
    Output: "Disease A; Disease B; Disease C"
    """
    if pd.isna(diagnosis_text):
        return diagnosis_text
    
    # Split by " also known as " first
    parts = re.split(r'\s+also\s+known\s+as\s+', diagnosis_text, flags=re.IGNORECASE)
    
    # Then split each part by " or "
    all_diseases = []
    for part in parts:
        sub_parts = re.split(r'\s+or\s+', part, flags=re.IGNORECASE)
        all_diseases.extend(sub_parts)
    
    # Clean up each disease name (strip whitespace and remove trailing punctuation)
    cleaned_diseases = []
    for disease in all_diseases:
        cleaned = disease.strip()
        # Remove trailing punctuation if present
        cleaned = cleaned.rstrip('.,;:')
        if cleaned:  # Only add non-empty strings
            cleaned_diseases.append(cleaned)
    
    # Join with semicolon and space
    return '; '.join(cleaned_diseases)

# Apply the transformation to the diagnosis column
df['diagnosis'] = df['diagnosis'].apply(format_diagnosis)

# Save the output
df.to_csv(os.path.join(script_dir, 'after.csv'), index=False)

print("✅ ETL completed successfully!")
print(f"   - Processed {len(df)} records")
print(f"   - Formatted diagnosis names in 'diagnosis' column")
print(f"   - Output saved to 'after.csv'")