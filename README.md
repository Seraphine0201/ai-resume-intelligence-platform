# AI Resume Intelligence Platform

## Overview

AI Resume Intelligence Platform is a Retrieval-Augmented Generation (RAG) application that analyzes resumes against job descriptions and provides ATS-style insights using Generative AI.

The system extracts resume content from PDF files, converts the content into vector embeddings, stores them in a vector database (ChromaDB), retrieves the most relevant resume sections based on a job description, and generates intelligent recommendations using Gemini 2.5 Flash.

---

## Features

✅ PDF Resume Upload

✅ Resume Text Extraction using PyMuPDF

✅ Semantic Text Chunking

✅ Vector Embeddings using Sentence Transformers

✅ ChromaDB Vector Database

✅ Retrieval-Augmented Generation (RAG)

✅ ATS Match Score Estimation

✅ Matching Skills Identification

✅ Missing Skills Detection

✅ Resume Improvement Suggestions

✅ Interview Question Generation

✅ Personalized Learning Roadmap

✅ Adjustable Temperature Control

✅ Adjustable Top-P Sampling

✅ Adjustable Top-K Sampling

---

## Architecture

```text
Resume PDF
    ↓
PyMuPDF Text Extraction
    ↓
Text Chunking
    ↓
Sentence Transformer Embeddings
    ↓
ChromaDB Vector Database
    ↓
Semantic Retrieval
    ↓
Top Relevant Resume Chunks
    ↓
Gemini 2.5 Flash
    ↓
ATS Analysis & Recommendations
```

---

## RAG Workflow

### Step 1: Resume Upload
The user uploads a PDF resume through the Streamlit interface.

### Step 2: Text Extraction
Resume content is extracted using PyMuPDF.

### Step 3: Chunking
The resume is split into smaller chunks using RecursiveCharacterTextSplitter.
```python
chunk_size=500
chunk_overlap=50
```

### Step 4: Embedding Generation
Each chunk is converted into vector embeddings using:
```python
all-MiniLM-L6-v2
```

### Step 5: Vector Storage
Embeddings are stored in ChromaDB.

### Step 6: Semantic Retrieval
The job description is converted into an embedding and used to retrieve the most relevant resume chunks.

### Step 7: LLM Analysis
Retrieved chunks are provided to Gemini 2.5 Flash for ATS-style analysis.

---

## Technologies Used

### Frontend

- Streamlit

### LLM

- Gemini 2.5 Flash

### RAG Components

- ChromaDB
- Sentence Transformers
- LangChain Text Splitters

### PDF Processing

- PyMuPDF

### Environment Management

- python-dotenv

### Programming Language

- Python

---

## Project Structure

```text
AI-Resume-Intelligence-Platform/
│
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
├── .env
└── venv/
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/your-username/ai-resume-intelligence-platform.git
```

### Navigate to Project Folder

```bash
cd ai-resume-intelligence-platform
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Create .env File

```env
GOOGLE_API_KEY=YOUR_API_KEY
```

### Run Application

```bash
streamlit run app.py
```

---

## Sample Output

The platform generates:

- ATS Match Score
- Matching Skills
- Missing Skills
- Resume Improvement Suggestions
- Interview Questions
- Learning Roadmap

---

## Hyperparameter Controls

The application allows users to dynamically tune LLM behavior through:

### Temperature

Controls response randomness.

Range:

```text
0.0 - 1.0
```

### Top-P

Controls nucleus sampling.

Range:

```text
0.0 - 1.0
```

### Top-K

Controls token candidate selection.

Range:

```text
1 - 100
```

---

## Future Enhancements

- HyDE Retrieval
- Multi-Query Retrieval
- Resume Chatbot
- Job Recommendation Engine
- Persistent ChromaDB Storage
- Resume Ranking System
- Streamlit Cloud Deployment
- Multi-Resume Comparison

---

## Author

Seraphine T

AI Engineer Aspirant | Generative AI | RAG Systems | Machine Learning

LinkedIn: www.linkedin.com/in/seraphine-t-346241250

GitHub: https://github.com/Seraphine0201