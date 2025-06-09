#!/usr/bin/env python3
"""
Minimal script to get structured diagnoses from JohnSnow Labs model
using TGI grammar constraints for guaranteed JSON output.
"""

import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def diagnose_with_severity(prompt: str) -> dict:
    """
    Get a single diagnosis with severity score from JohnSnow model.
    
    Args:
        prompt: Clinical description or symptoms
        
    Returns:
        Dict with diagnosis and severity (S0-S10)
    """
    # Environment setup
    hf_token = os.getenv("HF_TOKEN")
    endpoint_url = os.getenv("JONSNOW_ENDPOINT_URL")  # Should end with .hf.space
    
    if not hf_token or not endpoint_url:
        raise ValueError("Set HF_TOKEN and JONSNOW_ENDPOINT_URL in .env file")
    
    # The schema you want to enforce
    schema = {
        "type": "object",
        "properties": {
            "diagnosis": {
                "type": "string",
                "description": "The medical diagnosis being evaluated"
            },
            "severity": {
                "type": "string",
                "pattern": "^S(10|[0-9])$",
                "description": "Severity score from S0 to S10"
            }
        },
        "required": ["diagnosis", "severity"],
        "additionalProperties": False
    }
    
    # Build the request - this is the key difference
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 100,
            "temperature": 0.1,  # Low for consistent structure
            "grammar": {
                "type": "json",
                "value": schema  # TGI will enforce this schema
            }
        }
    }
    
    # Make the request to /generate endpoint
    response = requests.post(
        f"{endpoint_url}/generate",  # Note: /generate, not the base URL
        headers={
            "Authorization": f"Bearer {hf_token}",
            "Content-Type": "application/json"
        },
        json=payload
    )
    
    response.raise_for_status()
    
    # The response is already valid JSON matching our schema
    result = response.json()
    
    # TGI returns the generated text in a specific format
    if isinstance(result, dict) and "generated_text" in result:
        # Parse the generated JSON text
        return json.loads(result["generated_text"])
    else:
        # Some TGI configurations return the JSON directly
        return result


# Example usage
if __name__ == "__main__":
    # Test prompt
    clinical_case = """
    Patient: 45-year-old male presenting with:
    - Sudden onset chest pain radiating to left arm
    - Diaphoresis and nausea
    - BP 160/95, HR 110
    Provide diagnosis and severity.
    """
    
    try:
        result = diagnose_with_severity(clinical_case)
        print("Structured diagnosis received:")
        print(json.dumps(result, indent=2))
        
        # Access the structured data
        print(f"\nDiagnosis: {result['diagnosis']}")
        print(f"Severity: {result['severity']}")
        
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        print(f"Response: {e.response.text}")
    except Exception as e:
        print(f"Error: {e}")