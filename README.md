# 🚀 AI Document Intelligence System

> An AI-powered document analysis system developed during the **Tips Hindawi Internship (August–October 2026)** as part of the **Large Language Models (LLMs) Program**.

## 👤 Participant

| Field | Value |
|---|---|
| Full Name | Mohamed Mahmoud Ibrahim Ossaimee|
| Project Name | AI Document Intelligence System |
| GitHub Username | Momo19471 |
| Internship Batch | August–October 2026 |
| Training Program | Large Language Models (LLMs) Program |
| Organization | [Edrak for Ai](https://edrak4ai.com/en) |

---

## 📖 Project Overview

The **AI Document Intelligence System** is a Retrieval-Augmented Generation (RAG) application designed to interact with uploaded PDF documents.

The system extracts document content, divides it into meaningful text chunks, generates semantic embeddings, and stores them in a **FAISS vector database**. Relevant information is retrieved according to the user's query and provided as context to **Mistral-Nemo-Instruct-2407** for document-grounded generation.

The system supports multiple document intelligence tasks:

- Question Answering
- Document Summarization
- Structured Document Analysis
- Document Comparison

The project uses a **Streamlit** interface for interaction, while the Mistral language model can be hosted on a **Kaggle GPU** and exposed through a **FastAPI + ngrok** API.

### System Architecture

```text
                    PDF INPUT
                        │
                        ▼
                 PDF Extraction
                   PyPDFLoader
                        │
                        ▼
                  Text Chunking
                        │
                        ▼
                  Embeddings
              all-MiniLM-L6-v2
                        │
                        ▼
                    FAISS
                 Vector Store
                        │
                        ▼
                   Retriever
                        │
                        ▼
                Relevant Context
                        │
                        ▼
              Mistral-Nemo-Instruct
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
         Q&A         SUMMARY       ANALYSIS
          │             │             │
          └─────────────┴─────────────┘
                        │
                        ▼
                 Final Results
                        │
                        ▼
                 Streamlit UI
```

---

## ✨ Features

### ❓ Question Answering
- Upload a PDF document.
- Ask questions about its content.
- Retrieve the most relevant document chunks using semantic similarity.
- Generate answers using only the retrieved document context.
- Preserve document page references when available.

### 📝 Document Summarization
- Process the document in chunks.
- Generate summaries for individual sections.
- Combine the section summaries into a final document summary.
- Extract important facts, dates, numbers, names, and requirements.

### 📊 Structured Document Analysis
The system produces structured results containing:

- Document description
- Summary
- Key points
- Potential risks
- References

Structured output is generated using LangChain's `StructuredOutputParser`.

### 🔄 Document Comparison
Two PDF documents can be compared to identify:

- Overall changes
- Differences
- Added information
- Removed information
- Risks related to actual changes

The comparison prompt is designed to preserve exact values such as dates, amounts, durations, and conditions.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | User interface |
| Mistral-Nemo-Instruct-2407 | Large Language Model |
| Hugging Face Transformers | Model loading and inference |
| Sentence Transformers | Text embeddings |
| all-MiniLM-L6-v2 | Embedding model |
| FAISS | Vector database / similarity search |
| LangChain | Document processing and RAG components |
| PyPDF | PDF text extraction |
| FastAPI | LLM inference API |
| Uvicorn | API server |
| ngrok | Secure public tunnel to the API |
| Kaggle GPU | Remote model inference |

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd AI-Document-Intelligence-System
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install the required packages

```bash
pip install -r requirements.txt
```

### 4. Configure the model API

The Streamlit application communicates with the Mistral model through the FastAPI endpoint.

Set the API URL as an environment variable instead of hardcoding the temporary ngrok URL.

Example:

```env
MODEL_API_URL=https://YOUR-NGROK-URL.ngrok-free.dev/generate
```

**Do not commit `.env` or API credentials to GitHub.**

### 5. Start the Streamlit application

```bash
streamlit run app.py
```

---

## 🚀 Usage

### Question Answering

1. Start the Streamlit application.
2. Select **Question Answering**.
3. Upload a PDF document.
4. Enter a question.
5. Click **Ask Question**.
6. The system retrieves relevant document sections and generates an answer.

### Summarization

1. Select **Summarization**.
2. Upload a PDF document.
3. Start the summarization process.
4. The system generates section-level summaries and combines them into a final summary.

### Structured Analysis

1. Select **Structured Analysis**.
2. Upload a PDF document.
3. Run the analysis.
4. The system returns structured information including summary, key points, risks, and references.

### Document Comparison

1. Select **Document Comparison**.
2. Upload the two PDF documents.
3. Run the comparison.
4. Review the identified differences, additions, removals, and related risks.

---

## 📈 Results

The project demonstrates an end-to-end document intelligence workflow combining:

- PDF document processing
- Text chunking
- Semantic embeddings
- FAISS vector retrieval
- Retrieval-Augmented Generation
- Large Language Model inference
- Structured output parsing
- Multi-document comparison
- Interactive Streamlit visualization

The system provides a practical workflow for asking questions and extracting useful information from documents while grounding generated responses in the provided document content.

---

## 🔮 Future Improvements

- Add persistent vector database storage.
- Support additional document formats such as DOCX and TXT.
- Improve retrieval using hybrid search.
- Add reranking for more accurate context retrieval.
- Add conversation history for multi-turn document Q&A.
- Improve structured output validation.
- Add authentication and access control.
- Deploy the application using a permanent cloud inference endpoint instead of a temporary ngrok tunnel.
- Add automated evaluation metrics for RAG answer quality.

---

## 📚 About the Internship

This project was developed as part of the [**Tips Hindawi**](https://www.tipshindawi.com/) **Internship (August–October 2026)** and is intended to demonstrate practical application of Large Language Model technologies.

[**Tips Hindawi**](https://www.tipshindawi.com/) is the internships department of [**Edrak for Ai**](https://edrak4ai.com/en). The internship focuses on practical training, real-world projects, and showcasing participants' work through GitHub.

---

## 📄 License

This project is shared for educational and portfolio purposes.
