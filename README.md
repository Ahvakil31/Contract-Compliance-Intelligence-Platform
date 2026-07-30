# 📋 Contract Intelligence Platform

An AI-powered legal document analysis system that automatically extracts, classifies, and analyzes contract clauses using NLP and machine learning.

## 🚀 Features

- 📄 **Multi-format Support**: PDF, DOCX, TXT, and scanned images via OCR
- 🧠 **Intelligent Extraction**: Uses regex patterns, spaCy NER, and fine-tuned LegalBERT
- 📊 **5-Pillar Analysis**:
  - Core Profile & Metadata
  - Term & Exit Infrastructure
  - Financial & Operational Obligations
  - Legal Risk & Liability Allocation
  - Dispute & Venue Mechanics
- 🔄 **Asynchronous Processing**: Celery + Redis for background tasks
- 🗄️ **Vector Database**: Pinecone integration for semantic search
- 🐳 **Containerized**: Docker and Docker Compose for easy deployment

## 🛠️ Technology Stack

- **Backend**: FastAPI, Python 3.10+
- **NLP**: spaCy, Hugging Face Transformers (LegalBERT), PyTesseract
- **Task Queue**: Celery, Redis
- **Database**: Pinecone (vector), Redis (cache)
- **Container**: Docker, Docker Compose
- **Deployment**: GitHub Actions, Nginx

## 📁 Project Structure
