import subprocess
import sys

def install_packages():
    """Install all required packages for the AI ethics LLM project"""
    print("="*60)
    print("AI Ethics Education LLM - Dependency Installation")
    print("="*60)

    # Option 1: Install from requirements.txt
    print("\nInstalling from requirements.txt...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("\nAll packages installed successfully!")
    except subprocess.CalledProcessError:
        print("\nFailed to install from requirements.txt. Installing packages individually...")

        # Option 2: Install individual packages
        required_packages = [
            "torch>=2.0.0",
            "transformers>=4.40.0",
            "datasets>=2.18.0",
            "accelerate>=0.26.0",
            "peft>=0.10.0",
            "bitsandbytes>=0.43.0",
            "gradio>=4.0.0",
            "pandas>=2.0.0",
            "requests>=2.31.0",
            "beautifulsoup4>=4.12.0",
            "tqdm>=4.65.0"
        ]

        for package in required_packages:
            print(f"Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            except subprocess.CalledProcessError as e:
                print(f"Warning: Failed to install {package}. Error: {e}")

        print("\nPackage installation complete!")

    print("\n" + "="*60)
    print("Setup Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run 'python data_collection.py' to prepare training data")
    print("2. Run 'python model_training_qlora.py' to fine-tune the model")
    print("3. Run 'python evaluation_rubric.py' to evaluate the model")
    print("4. Run 'python app.py' to launch the Gradio interface")
    print("="*60)

if __name__ == "__main__":
    install_packages() 