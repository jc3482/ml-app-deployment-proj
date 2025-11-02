"""
Hugging Face Spaces version of SmartPantry app.
This file should be copied to the root as 'app.py' for Spaces deployment.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from app.main import SmartPantryApp

def main():
    """Entry point for Hugging Face Spaces."""
    # Initialize app with Spaces-optimized config
    app = SmartPantryApp(config_path="config.yaml")
    
    # Launch with Spaces settings
    app.launch(
        share=False,  # Spaces provides its own sharing
        server_name="0.0.0.0",
        server_port=7860,
    )

if __name__ == "__main__":
    main()

