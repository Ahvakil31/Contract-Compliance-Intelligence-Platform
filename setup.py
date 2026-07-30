# setup.py
import subprocess
import sys
import os

def setup_environment():
    """Setup the environment for the legal NLP platform."""
    print("🚀 Setting up Legal NLP Platform...")
    
    # Install dependencies
    print("📦 Installing Python dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
    
    # Download spaCy model
    print("📥 Downloading spaCy model...")
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
    
    # Create directories
    print("📁 Creating directories...")
    os.makedirs("static", exist_ok=True)
    os.makedirs("/tmp/legal_ingestion_vault", exist_ok=True)
    
    # Copy index.html to static folder
    if os.path.exists("index.html"):
        import shutil
        shutil.copy("index.html", "static/index.html")
    
    print("✅ Setup complete!")
    print("\nTo start the application:")
    print("  docker-compose up -d")
    print("  or")
    print("  uvicorn main:app --host 0.0.0.0 --port 8000 --reload")

if __name__ == "__main__":
    setup_environment()