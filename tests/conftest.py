import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Tests must be deterministic and offline: never let .env route them to a cloud LLM.
os.environ["RETURNIQ_LLM"] = "rules"
for _k in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "NVIDIA_API_KEY", "OLLAMA_API_KEY"):
    os.environ.pop(_k, None)
