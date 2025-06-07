#!/usr/bin/env python3
"""
run2.py - CSV to JSON Transformation with Diagnosis Severity Mapping

This script:
1. Reads the original CSV files (ramedis_before.csv and urg_torre_before.csv)
2. Loads the severity mapping from diagnosis_severity_mapping.csv
3. Transforms the diagnosis column from "diagnosis1; diagnosis2; diagnosis3" format
4. Creates nested JSON structure with individual diagnoses and their severity scores
5. For ramedis.csv, also handles diagnostic_code nesting
6. Outputs ramedis.json and urg_torre.json
7. Filters out cases with S0 severity and renumbers IDs

JSON Structure:
{
  "id": 123,
  "case": "...",
  "diagnoses": [
    {
      "name": "diagnosis_name",
      "severity": "S5"
    },
    ...
  ],
  "diagnostic_codes": [...],  // Only for ramedis
  ...other_fields
}
"""

import pandas as pd
import json
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configuration
RAMEDIS_INPUT = "ramedis.csv"
URG_TORRE_INPUT = "urg_torre.csv"
SEVERITY_MAPPING_FILE = "diagnosis_severity_mapping.csv"
RAMEDIS_OUTPUT = "ramedis.json"
URG_TORRE_OUTPUT = "urg_torre.json"

def load_severity_mapping(file_path: str) -> Dict[str, str]:
    """Load the diagnosis-to-severity mapping from CSV."""
    print(f"📚 Loading severity mapping from {file_path}...")
    
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='latin-1')
    
    # Create mapping dictionary
    mapping = dict(zip(df['g_diagnosis'], df['severity']))
    
    print(f"✅ Loaded {len(mapping)} diagnosis-severity mappings")
    return mapping

def parse_diagnostic_codes(diagnostic_code_str: str) -> List[str]:
    """Parse diagnostic codes from string format like '[OMIM:251000, ORPHA:27, CCRD:71]'."""
    if pd.isna(diagnostic_code_str) or diagnostic_code_str.strip() == "":
        return []
    
    try:
        # Handle string representation of list
        if diagnostic_code_str.startswith('[') and diagnostic_code_str.endswith(']'):
            # Remove brackets and split by comma
            codes_str = diagnostic_code_str[1:-1]
            codes = [code.strip() for code in codes_str.split(',')]
            return [code for code in codes if code]
        else:
            # Single code or different format
            return [diagnostic_code_str.strip()]
    except Exception:
        return []

def process_diagnoses_with_severity(diagnosis_str: str, severity_mapping: Dict[str, str]) -> List[Dict[str, str]]:
    """Process diagnosis string and attach severity scores."""
    if pd.isna(diagnosis_str) or diagnosis_str.strip() == "":
        return []
    
    # Split by "; " and clean up
    individual_diagnoses = [d.strip() for d in str(diagnosis_str).split(";")]
    individual_diagnoses = [d for d in individual_diagnoses if d and d != ""]
    
    # Create list of diagnosis objects with severity
    diagnoses_with_severity = []
    for diagnosis in individual_diagnoses:
        severity = severity_mapping.get(diagnosis, "S5")  # Default to S5 if not found
        diagnoses_with_severity.append({
            "name": diagnosis,
            "severity": severity
        })
    
    return diagnoses_with_severity

def has_s0_severity(diagnoses: List[Dict[str, str]]) -> bool:
    """Check if any diagnosis has S0 severity."""
    return any(diag['severity'] == 'S0' for diag in diagnoses)

def transform_ramedis_to_json(csv_file: str, severity_mapping: Dict[str, str]) -> List[Dict[str, Any]]:
    """Transform ramedis CSV to JSON format with nested diagnoses and diagnostic codes."""
    print(f"🔄 Transforming {csv_file} to JSON format...")
    
    # Load CSV
    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_file, encoding='latin-1')
    
    json_records = []
    new_id = 1
    
    for idx, row in df.iterrows():
        # Process diagnoses with severity
        diagnoses = process_diagnoses_with_severity(row.get('diagnosis', ''), severity_mapping)
        
        # Skip records with S0 severity
        if has_s0_severity(diagnoses):
            continue
        
        # Process diagnostic codes
        diagnostic_codes = parse_diagnostic_codes(row.get('diagnostic_code', ''))
        
        # Create JSON record with new ID
        record = {
            "id": new_id,
            "case": str(row['case']) if pd.notna(row['case']) else "",
            "complexity": str(row['complexity']) if pd.notna(row['complexity']) else "",
            "diagnoses": diagnoses,
            "diagnostic_codes": diagnostic_codes,
            "origin": str(row['origin']) if pd.notna(row['origin']) else ""
        }
        
        json_records.append(record)
        new_id += 1
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1} records...")
    
    print(f"✅ Transformed {len(json_records)} records from {csv_file}")
    return json_records

def transform_urg_torre_to_json(csv_file: str, severity_mapping: Dict[str, str]) -> List[Dict[str, Any]]:
    """Transform urg_torre CSV to JSON format with nested diagnoses."""
    print(f"🔄 Transforming {csv_file} to JSON format...")
    
    # Load CSV (handle large file)
    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_file, encoding='latin-1')
    
    json_records = []
    new_id = 1
    
    for idx, row in df.iterrows():
        # Process diagnoses with severity
        diagnoses = process_diagnoses_with_severity(row.get('diagnosis', ''), severity_mapping)
        
        # Skip records with S0 severity
        if has_s0_severity(diagnoses):
            continue
        
        # Create JSON record with new ID
        record = {
            "id": new_id,
            "case": str(row['case']) if pd.notna(row['case']) else "",
            "complexity": str(row['complexity']) if pd.notna(row['complexity']) else "",
            "diagnoses": diagnoses,
            "original_diagnosis": str(row['original_diagnosis']) if pd.notna(row['original_diagnosis']) else "",
            "icd10_diagnosis": str(row['icd10_diagnosis']) if pd.notna(row['icd10_diagnosis']) else "",
            "diagnostic_code": str(row['diagnostic_code']) if pd.notna(row['diagnostic_code']) else "",
            "death": int(row['death']) if pd.notna(row['death']) else 0,
            "critical": int(row['critical']) if pd.notna(row['critical']) else 0,
            "pediatric": int(row['pediatric']) if pd.notna(row['pediatric']) else 0,
            "severity": int(row['severity']) if pd.notna(row['severity']) else 0
        }
        
        json_records.append(record)
        new_id += 1
        
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1} records...")
    
    print(f"✅ Transformed {len(json_records)} records from {csv_file}")
    return json_records

def save_json(data: List[Dict[str, Any]], output_file: str):
    """Save data to JSON file with proper encoding."""
    print(f"💾 Saving {len(data)} records to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Saved to {output_file}")

def validate_severity_coverage(severity_mapping: Dict[str, str], diagnoses_used: set):
    """Validate how many diagnoses from the files are covered by the severity mapping."""
    covered = len(diagnoses_used.intersection(set(severity_mapping.keys())))
    total = len(diagnoses_used)
    coverage_pct = (covered / total * 100) if total > 0 else 0
    
    print(f"\n📊 Severity Mapping Coverage:")
    print(f"  Diagnoses in files: {total}")
    print(f"  Diagnoses with severity mapping: {covered}")
    print(f"  Coverage: {coverage_pct:.1f}%")
    
    if coverage_pct < 95:
        print(f"  ⚠️ Warning: {total - covered} diagnoses will use default severity S5")

def main():
    """Main execution function."""
    print("🔄 Starting CSV to JSON Transformation Pipeline")
    print("=" * 60)
    
    # Step 1: Load severity mapping
    print("📚 Step 1: Loading severity mapping")
    severity_mapping = load_severity_mapping(SEVERITY_MAPPING_FILE)
    
    # Step 2: Transform ramedis.csv
    print(f"\n🏥 Step 2: Processing {RAMEDIS_INPUT}")
    ramedis_json = transform_ramedis_to_json(RAMEDIS_INPUT, severity_mapping)
    
    # Step 3: Transform urg_torre.csv
    print(f"\n🚑 Step 3: Processing {URG_TORRE_INPUT}")
    urg_torre_json = transform_urg_torre_to_json(URG_TORRE_INPUT, severity_mapping)
    
    # Step 4: Collect all diagnoses used for validation
    print("\n🔍 Step 4: Validating severity coverage")
    all_diagnoses_used = set()
    
    # Collect from ramedis
    for record in ramedis_json:
        for diag in record['diagnoses']:
            all_diagnoses_used.add(diag['name'])
    
    # Collect from urg_torre
    for record in urg_torre_json:
        for diag in record['diagnoses']:
            all_diagnoses_used.add(diag['name'])
    
    validate_severity_coverage(severity_mapping, all_diagnoses_used)
    
    # Step 5: Save JSON files
    print(f"\n💾 Step 5: Saving JSON files")
    save_json(ramedis_json, RAMEDIS_OUTPUT)
    save_json(urg_torre_json, URG_TORRE_OUTPUT)
    
    # Summary
    print("\n📈 Summary:")
    print(f"  RAMEDIS records: {len(ramedis_json)}")
    print(f"  URG_TORRE records: {len(urg_torre_json)}")
    print(f"  Total diagnoses with severity: {len(all_diagnoses_used)}")
    
    # Sample output structure
    print(f"\n📄 Sample JSON structure (ramedis):")
    if ramedis_json:
        sample = {k: v for k, v in ramedis_json[0].items() if k != 'case'}
        sample['case'] = "... (truncated) ..."
        print(json.dumps(sample, indent=2, ensure_ascii=False)[:500] + "...")
    
    print(f"\n✅ Transformation pipeline completed successfully!")
    print(f"📁 Output files: {RAMEDIS_OUTPUT}, {URG_TORRE_OUTPUT}")

if __name__ == "__main__":
    main() 