#!/usr/bin/env python3
"""
run1.py - Diagnosis Severity Assessment using LLM Batch Processing

This script:
1. Extracts all diagnoses from ramedis.csv and urg_torre.csv (diagnosis column)
2. Splits by "; " and creates a unique set of all diseases
3. Shuffles the unique diagnoses
4. Uses utils.llm in batch mode (N=100) to assess severity/rarity (S0-S10)
5. Saves results to CSV with columns: g_diagnosis, severity

Severity Scale:
- S0-S2: Minor conditions, easy to cure, not life-threatening
- S3-S5: Moderate conditions, some risk or complexity
- S6-S7: Serious conditions, significant health impact
- S8-S10: Severe/rare conditions, high mortality, very complex treatment
"""

import pandas as pd
import random
from pathlib import Path
from typing import List, Set
from utils.llm import Azure
import json

# Configuration
BATCH_SIZE = 100
RAMEDIS_FILE = "ramedis.csv"
URG_TORRE_FILE = "urg_torre.csv" 
OUTPUT_FILE = "diagnosis_severity_mapping.csv"

def extract_diagnoses_from_file(file_path: str, diagnosis_column: str) -> Set[str]:
    """Extract unique diagnoses from a CSV file."""
    print(f"📁 Processing {file_path}...")
    
    # Load CSV with proper encoding handling
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='latin-1')
    
    diagnoses = set()
    
    # Process diagnosis column
    for idx, diagnosis_str in enumerate(df[diagnosis_column].dropna()):
        if pd.isna(diagnosis_str) or diagnosis_str.strip() == "":
            continue
            
        # Split by "; " and clean up
        individual_diagnoses = [d.strip() for d in str(diagnosis_str).split(";")]
        diagnoses.update([d for d in individual_diagnoses if d and d != ""])
        
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1} rows, found {len(diagnoses)} unique diagnoses so far")
    
    print(f"  ✅ Found {len(diagnoses)} unique diagnoses from {file_path}")
    return diagnoses

def assess_severity_batch(diagnoses_batch: List[str], llm: Azure) -> List[dict]:
    """Assess severity for a batch of diagnoses using LLM."""
    
    # Prepare batch items for LLM processing
    batch_items = [{"diagnosis": diagnosis} for diagnosis in diagnoses_batch]
    
    # Define the prompt for severity assessment
    prompt = """
    You are a medical expert evaluating disease severity and rarity. For each diagnosis provided, assign a severity score from S0 to S10 based on these criteria:

    **Severity Scale:**
    - S0-S2: Minor conditions (common colds, minor cuts, mild headaches, minor fractures)
    - S3-S5: Moderate conditions (pneumonia, diabetes complications, moderate infections)
    - S6-S7: Serious conditions (heart attacks, strokes, major surgeries needed)
    - S8-S10: Severe/rare conditions (rare genetic diseases, terminal cancers, complex multi-organ failures)

    **Consider these factors:**
    - Mortality risk
    - Treatment complexity
    - Disease rarity
    - Long-term health impact
    - Urgency of medical intervention required

    For each diagnosis, return ONLY the severity code (S0, S1, S2, ..., S10) based on the medical severity and rarity.
    """
    
    # Define schema for structured output
    schema = {
        "type": "object",
        "properties": {
            "diagnosis": {"type": "string"},
            "severity": {"type": "string", "pattern": "^S(0|[1-9]|10)$"}
        },
        "required": ["diagnosis", "severity"]
    }
    
    try:
        # Call LLM with batch processing
        results = llm.generate(
            prompt,
            batch_items=batch_items,
            schema=schema,
            temperature=0.3  # Low temperature for consistency
        )
        
        return results if isinstance(results, list) else []
        
    except Exception as e:
        print(f"  ⚠️ Error in LLM batch processing: {e}")
        # Fallback: assign default severity
        return [{"diagnosis": d, "severity": "S5"} for d in diagnoses_batch]

def process_diagnoses_with_llm(diagnoses: List[str]) -> pd.DataFrame:
    """Process all diagnoses with LLM in batches."""
    print(f"🤖 Processing {len(diagnoses)} diagnoses with LLM (batch size: {BATCH_SIZE})...")
    
    # Initialize LLM
    llm = Azure("gpt-4o")
    
    all_results = []
    
    # Process in batches
    for i in range(0, len(diagnoses), BATCH_SIZE):
        batch = diagnoses[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (len(diagnoses) + BATCH_SIZE - 1) // BATCH_SIZE
        
        print(f"  Processing batch {batch_num}/{total_batches} ({len(batch)} items)...")
        
        batch_results = assess_severity_batch(batch, llm)
        all_results.extend(batch_results)
        
        print(f"  ✅ Completed batch {batch_num}/{total_batches}")
    
    # Convert to DataFrame
    df_results = pd.DataFrame(all_results)
    
    # Rename columns to match expected output
    df_results = df_results.rename(columns={"diagnosis": "g_diagnosis"})
    
    print(f"✅ LLM processing completed. Generated {len(df_results)} severity assessments")
    return df_results

def main():
    """Main execution function."""
    print("🏥 Starting Diagnosis Severity Assessment Pipeline")
    print("=" * 60)
    
    # Step 1: Extract diagnoses from both files
    print("📊 Step 1: Extracting diagnoses from CSV files")
    
    ramedis_diagnoses = extract_diagnoses_from_file(RAMEDIS_FILE, "diagnosis")
    urg_torre_diagnoses = extract_diagnoses_from_file(URG_TORRE_FILE, "diagnosis")
    
    # Combine and create unique set
    all_diagnoses = ramedis_diagnoses.union(urg_torre_diagnoses)
    print(f"🔄 Combined unique diagnoses: {len(all_diagnoses)}")
    
    # Step 2: Shuffle diagnoses for random processing order
    print("\n🔀 Step 2: Shuffling diagnoses for random processing")
    diagnoses_list = list(all_diagnoses)
    random.shuffle(diagnoses_list)
    print(f"✅ Shuffled {len(diagnoses_list)} unique diagnoses")
    
    # Step 3: Process with LLM in batches
    print(f"\n🤖 Step 3: Processing with LLM (batches of {BATCH_SIZE})")
    severity_df = process_diagnoses_with_llm(diagnoses_list)
    
    # Step 4: Save results
    print(f"\n💾 Step 4: Saving results to {OUTPUT_FILE}")
    severity_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    
    # Summary
    print("\n📈 Summary:")
    print(f"  Total unique diagnoses processed: {len(severity_df)}")
    print(f"  Severity distribution:")
    severity_counts = severity_df['severity'].value_counts().sort_index()
    for severity, count in severity_counts.items():
        print(f"    {severity}: {count} diagnoses")
    
    print(f"\n✅ Pipeline completed successfully!")
    print(f"📄 Results saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main() 