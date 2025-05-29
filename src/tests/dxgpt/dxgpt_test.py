from pathlib import Path
import sys

# Add parent directory to Python path to import ease_llm_client
current_dir = Path(__file__).parent
easellm_dir = current_dir.parent.parent / "EaseLLM"
sys.path.insert(0, str(easellm_dir))

from ease_llm_client import EaseLLM

# Initialize EaseLLM with Azure GPT-4
llm = EaseLLM(provider="azure", model="gpt-4o")

# Path to dxgpt prompt config
config_path = current_dir / "dxgpt_prompt.yaml"

print(llm.call(config_path, description="Dolor abdominal fortísimo"))
