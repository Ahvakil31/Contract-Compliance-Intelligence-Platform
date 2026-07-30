@echo off
echo 🚀 Starting Legal NLP Platform...

echo 📥 Installing dependencies...
pip install -r requirements.txt

echo 📥 Downloading spaCy model...
python -m spacy download en_core_web_sm

echo 📁 Creating directories...
mkdir static 2>nul
mkdir C:\tmp\legal_ingestion_vault 2>nul
copy index.html static\index.html 2>nul

echo ✅ Setup complete!

echo 🚀 Starting services...
start /B docker-compose up -d

echo ⏳ Waiting for services to start...
timeout /t 10

echo 🌐 Open browser at http://localhost:8000/ui
start http://localhost:8000/ui

echo ✅ All services running!