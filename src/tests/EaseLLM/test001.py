#!/usr/bin/env python3
"""
test001.py
─────────
Minimal test to demonstrate EaseLLM functionality

Requirements:
- Set up environment variables for your chosen provider
- For Azure: AZURE_{MODEL}_API_KEY, AZURE_{MODEL}_ENDPOINT, etc.
- For HuggingFace: HF_API_KEY, HF_ENDPOINT, HF_MODEL

Usage:
    python test001.py
"""

import os
import sys
from pathlib import Path

# Add current directory to Python path to import ease_llm_client
current_dir = Path(__file__).parent
# Add the parent directory containing the EaseLLM module
easellm_dir = current_dir.parent.parent / "EaseLLM"
sys.path.insert(0, str(easellm_dir))

from ease_llm_client import EaseLLM

def test_azure():
    """Test with Azure OpenAI"""
    print("🔵 Testing Azure OpenAI...")
    try:
        # Initialize EaseLLM with Azure provider
        llm = EaseLLM(provider="azure", model="gpt-4o")
        
        # Path to our test YAML config
        config_path = current_dir / "test001_config.yaml"
        
        # Test with default prompt
        print("📝 Testing default prompt...")
        result = llm.call(
            yaml_path=str(config_path),
            name="Alice",
            style="casual"
        )
        print(f"✅ Result: {result}")
        
        # Test with formal prompt
        print("📝 Testing formal prompt...")
        result_formal = llm.call(
            yaml_path=str(config_path),
            prompt_id="formal",
            name="Dr. Smith",
            context="business meeting"
        )
        print(f"✅ Formal result: {result_formal}")
        
        return True
        
    except Exception as e:
        print(f"❌ Azure test failed: {e}")
        return False

def test_huggingface():
    """Test with HuggingFace"""
    print("\n🤗 Testing HuggingFace...")
    try:
        # Initialize EaseLLM with HuggingFace provider
        llm = EaseLLM(provider="huggingface", model="jsl")
        
        # Path to our test YAML config
        config_path = current_dir / "test001_config.yaml"
        
        # Test with default prompt
        print("📝 Testing HuggingFace prompt...")
        result = llm.call(
            yaml_path=str(config_path),
            name="Bob",
            style="friendly"
        )
        print(f"✅ HF Result: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ HuggingFace test failed: {e}")
        return False

def main():
    print("\n🚀 Running EaseLLM tests...")
    
    # Run Azure test
    azure_success = test_azure()
    
    # Run HuggingFace test
    hf_success = test_huggingface()
    
    # Print final results
    print("\n📊 Test Results:")
    print(f"Azure: {'✅' if azure_success else '❌'}")
    print(f"HuggingFace: {'✅' if hf_success else '❌'}")
    
    # Return overall success
    return azure_success and hf_success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)