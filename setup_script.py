#!/usr/bin/env python3
"""
Professional ATS Resume Analyzer - Setup Script
Automated installation and configuration
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, check=True, shell=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}:")
        print(f"Command: {command}")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_requirements():
    """Install Python packages"""
    print("\n📦 Installing Python Dependencies...")
    
    # Core packages
    packages = [
        "pymupdf==1.23.14",
        "spacy==3.7.2", 
        "scikit-learn==1.3.2",
        "sentence-transformers==2.2.2",
        "skillner==1.0.5",
        "pandas==2.1.4",
        "numpy==1.24.4",
        "nltk==3.8.1",
        "python-dateutil==2.8.2",
        "regex==2023.10.3"
    ]
    
    for package in packages:
        if not run_command(f"pip install {package}", f"Installing {package.split('==')[0]}"):
            return False
    
    return True

def download_models():
    """Download required NLP models"""
    print("\n🧠 Downloading NLP Models...")
    
    # Download spaCy model
    if not run_command("python -m spacy download en_core_web_lg", 
                      "Downloading spaCy English model"):
        print("⚠️  Trying alternative spaCy model...")
        run_command("python -m spacy download en_core_web_sm", 
                   "Downloading spaCy English model (small)")
    
    # Download NLTK data
    nltk_command = """python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')" """
    run_command(nltk_command, "Downloading NLTK data")
    
    return True

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating Project Structure...")
    
    directories = [
        "examples",
        "reports", 
        "logs",
        "tests"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def create_example_files():
    """Create example configuration files"""
    print("\n📄 Creating Example Files...")
    
    # Create example job description
    example_jd = """
SOFTWARE ENGINEER - PYTHON DEVELOPER

We are seeking a talented Software Engineer with strong Python programming skills 
to join our dynamic development team.

REQUIREMENTS:
• Bachelor's degree in Computer Science or related field
• 3+ years of Python development experience
• Experience with web frameworks (Django, Flask)
• Knowledge of databases (SQL, PostgreSQL)
• Familiarity with version control (Git)
• Understanding of APIs and REST services
• Experience with cloud platforms (AWS, Azure)
• Strong problem-solving and analytical skills
• Excellent communication and teamwork abilities

PREFERRED SKILLS:
• Machine learning experience
• Docker and containerization
• CI/CD pipelines
• Agile development methodologies
• JavaScript and frontend frameworks
    """
    
    with open("examples/sample_job_description.txt", "w") as f:
        f.write(example_jd.strip())
    
    print("✅ Created sample job description")

def run_test():
    """Run a basic test to ensure everything works"""
    print("\n🧪 Running Basic Test...")
    
    test_code = '''
try:
    import spacy
    import sklearn
    import sentence_transformers
    from src import utils
    print("✅ All modules imported successfully!")
    
    # Test spaCy model
    try:
        nlp = spacy.load("en_core_web_lg")
        print("✅ spaCy large model loaded successfully!")
    except OSError:
        nlp = spacy.load("en_core_web_sm")
        print("✅ spaCy small model loaded successfully!")
    
    print("✅ Setup validation completed!")
    
except Exception as e:
    print(f"❌ Error during validation: {e}")
    '''
    
    return run_command(f'python -c "{test_code}"', "Validating installation")

def main():
    """Main setup function"""
    print("🚀 PROFESSIONAL ATS RESUME ANALYZER - SETUP")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements. Please check your internet connection.")
        sys.exit(1)
    
    # Download models
    download_models()
    
    # Create directories
    create_directories()
    
    # Create example files
    create_example_files()
    
    # Run validation test
    if run_test():
        print("\n🎉 SETUP COMPLETED SUCCESSFULLY!")
        print("\n📋 Next Steps:")
        print("1. Place your PDF resume in the project directory")
        print("2. Run: AI_Resume_Analyzer.bat")
        print("3. Follow the interactive prompts")
        print("\n🔗 For help: https://github.com/nik2401/AI-Powered-ATS-Resume-Analyzer")
    else:
        print("\n⚠️  Setup completed with some issues.")
        print("The application should still work, but some features may be limited.")
        print("Check the error messages above for details.")

if __name__ == "__main__":
    main()