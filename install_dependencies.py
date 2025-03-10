import subprocess
import sys

def install_packages():
    """Install all required packages for the AI ethics LLM project"""
    required_packages = [
        "pandas",
        "requests",
        "beautifulsoup4",
        "torch",
        "transformers",
        "datasets",
        "gradio",
        "accelerate>=0.26.0"
    ]
    
    print("Installing required packages...")
    for package in required_packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("All packages installed successfully!")

if __name__ == "__main__":
    install_packages() 