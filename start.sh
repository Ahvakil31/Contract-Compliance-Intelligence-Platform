#!/bin/bash
echo "🚀 Starting Legal NLP Platform..."

echo "📥 Installing dependencies..."
pip install -r requirements.txt

echo "📥 Downloading spaCy model..."
python -m spacy download en_core_web_sm

echo "📁 Creating directories..."
mkdir -p static
mkdir -p /tmp/legal_ingestion_vault
cp index.html static/index.html 2>/dev/null || true

echo "✅ Setup complete!"

echo "🚀 Starting services..."
docker-compose up -d

echo "⏳ Waiting for services to start..."
sleep 10

echo "🌐 Open browser at http://localhost:8000/ui"
open http://localhost:8000/ui 2>/dev/null || xdg-open http://localhost:8000/ui 2>/dev/null || echo "Open http://localhost:8000/ui in your browser"

echo "✅ All services running!"