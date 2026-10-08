import os
import re
import tempfile
import requests

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_classic.output_parsers import (
    StructuredOutputParser,
    ResponseSchema
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_NAME = "mistralai/Mistral-Nemo-Instruct-2407"

# Kaggle + FastAPI + ngrok
MODEL_API_URL = "MODEL_API_URL"

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# 2. LOCAL MODEL / CACHE SETTINGS
# ============================================================

# Only the embedding model is stored locally.
# Mistral is running remotely on Kaggle.

EMBEDDING_CACHE_DIR = r"./embedding_cache"

os.makedirs(
    EMBEDDING_CACHE_DIR,
    exist_ok=True
)


# ============================================================
# 3. OTHER SETTINGS
# ============================================================

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 3


# ============================================================
# 4. LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        cache_folder=EMBEDDING_CACHE_DIR
    )

    return embeddings


embedding_model = load_embedding_model()


# ============================================================
# 5. TEXT GENERATION
# ============================================================

def generate_text(
    prompt,
    max_new_tokens=300
):

    try:

        response = requests.post(
            MODEL_API_URL,
            json={
                "prompt": prompt,
                "max_new_tokens": max_new_tokens
            },
            timeout=300
        )

        response.raise_for_status()

        data = response.json()

        if "response" not in data:
            raise ValueError(
                "API response does not contain 'response'."
            )

        return data["response"]

    except requests.exceptions.Timeout:

        raise RuntimeError(
            "The Kaggle model request timed out."
        )

    except requests.exceptions.ConnectionError:

        raise RuntimeError(
            "Could not connect to the Kaggle Mistral API. "
            "Make sure the Kaggle notebook and ngrok tunnel "
            "are still running."
        )

    except requests.exceptions.RequestException as e:

        raise RuntimeError(
            f"Model API request failed: {e}"
        )


# ============================================================
# 6. PDF LOADING
# ============================================================

def load_pdf(pdf_path):

    if not os.path.isfile(pdf_path):

        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    loader = PyPDFLoader(
        pdf_path
    )

    documents = loader.load()

    if not documents:

        raise ValueError(
            "No content could be extracted from the PDF."
        )

    if not any(
        doc.page_content.strip()
        for doc in documents
    ):

        raise ValueError(
            "The PDF contains no extractable text."
        )

    return documents


# ============================================================
# 7. CHUNKING
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)


# ============================================================
# 8. BUILD CONTEXT
# ============================================================

def build_context(docs):

    context_parts = []

    for doc in docs:

        page = doc.metadata.get(
            "page",
            "Unknown"
        )

        if isinstance(page, int):

            page_number = page + 1

        else:

            page_number = page

        context_parts.append(
            f"[Page {page_number}]\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(
        context_parts
    )


# ============================================================
# 9. QUESTION ANSWERING
# ============================================================

qa_template = """
You are a document analysis assistant.

Answer the question using ONLY the provided context.

IMPORTANT RULES:

1. Do not use outside knowledge.

2. Do not invent information.

3. If the answer cannot be found in the context,
   say that the information is not available in the document.

4. Do NOT invent page numbers.

5. If the question asks for page numbers, use ONLY the
   [Page X] labels provided in the context.

6. If the question asks for a number, date, amount, name,
   or condition, preserve it exactly as stated in the context.

Context:
{context}

Question:
{question}

Answer:
"""


def answer_question(
    vectordb,
    question,
    k=TOP_K
):

    docs = vectordb.similarity_search(
        question,
        k=k
    )

    context = build_context(
        docs
    )

    prompt = qa_template.format(
        context=context,
        question=question
    )

    answer = generate_text(
        prompt,
        max_new_tokens=300
    )

    source_pages = sorted(
        set(
            doc.metadata["page"] + 1
            for doc in docs
            if isinstance(
                doc.metadata.get("page"),
                int
            )
        )
    )

    return {
        "answer": answer,
        "sources": docs,
        "source_pages": source_pages
    }


# ============================================================
# 10. SUMMARIZATION
# ============================================================

def summarize_chunks(chunks):

    summaries = []

    for chunk in chunks:

        prompt = f"""
You are a document summarization assistant.

Summarize the following document section.

Keep important:

- facts
- numbers
- dates
- names
- requirements
- conditions

Do not invent information.

Document section:

{chunk.page_content}

Summary:
"""

        summary = generate_text(
            prompt,
            max_new_tokens=200
        )

        summaries.append(
            summary
        )

    return summaries


def summarize_document(chunks):

    chunk_summaries = summarize_chunks(
        chunks
    )

    combined_summary = "\n\n".join(
        chunk_summaries
    )

    prompt = f"""
You are an intelligent document analysis assistant.

Create a final summary from the following
document section summaries.

Include:

1. Concise overall summary

2. Important key points

3. Potential risks

Use ONLY the provided information.

Do not invent information.

Section summaries:

{combined_summary}

Final summary:
"""

    return generate_text(
        prompt,
        max_new_tokens=500
    )


# ============================================================
# 11. STRUCTURED ANALYSIS
# ============================================================

document_schema = ResponseSchema(
    name="document",
    description=(
        "Name or description of the analyzed document."
    )
)

summary_schema = ResponseSchema(
    name="summary",
    description=(
        "Concise summary of the document."
    )
)

key_points_schema = ResponseSchema(
    name="key_points",
    description=(
        "Important points extracted from the document."
    )
)

risks_schema = ResponseSchema(
    name="risks",
    description=(
        "Potential risks or concerns identified "
        "in the document."
    )
)

references_schema = ResponseSchema(
    name="references",
    description=(
        "Relevant page numbers or document sections."
    )
)


response_schemas = [
    document_schema,
    summary_schema,
    key_points_schema,
    risks_schema,
    references_schema
]


output_parser = (
    StructuredOutputParser
    .from_response_schemas(
        response_schemas
    )
)


format_instructions = (
    output_parser.get_format_instructions()
)


analysis_template = """
You are an intelligent document analysis assistant.

Analyze the provided document context.

Use ONLY the information contained in the context.

Do not invent information.

Context:

{context}

{format_instructions}

Return ONLY the structured output.
"""


def extract_json_block(text):

    # Look for JSON inside ```json ... ```
    match = re.search(
        r"```json\s*(.*?)\s*```",
        text,
        re.DOTALL
    )

    if match:

        return match.group(1)

    # Look for a normal JSON object
    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL
    )

    if match:

        return match.group(0)

    raise ValueError(
        "No JSON object found in model response."
    )


def analyze_document(
    pdf_path,
    operation="qa",
    question=None
):

    documents = load_pdf(
        pdf_path
    )

    chunks = text_splitter.split_documents(
        documents
    )

    vectordb = FAISS.from_documents(
        chunks,
        embedding_model
    )

    if operation == "qa":

        if not question:

            raise ValueError(
                "Question is required for Q&A."
            )

        return answer_question(
            vectordb,
            question
        )

    elif operation == "summary":

        return summarize_document(
            chunks
        )

    elif operation == "analysis":

        context = build_context(
            chunks
        )

        prompt = analysis_template.format(
            context=context,
            format_instructions=format_instructions
        )

        response = generate_text(
            prompt,
            max_new_tokens=500
        )

        json_text = extract_json_block(
            response
        )

        return output_parser.parse(
            json_text
        )

    else:

        raise ValueError(
            f"Unsupported operation: {operation}"
        )


# ============================================================
# 12. DOCUMENT COMPARISON
# ============================================================

comparison_template = """
You are a precise document comparison assistant.

Your task is to compare Document A and Document B using ONLY
the exact information provided in the two documents.

IMPORTANT RULES:

1. Do not invent information.

2. Do not use outside knowledge.

3. Do not assume that information was removed just because it
   is summarized differently.

4. Only report something as ADDED if it appears in Document B
   and does not appear in Document A.

5. Only report something as REMOVED if it appears in Document A
   and does not appear in Document B.

6. If a value, date, amount, duration, number, or condition
   changes, explicitly show BOTH the old value and the new value.

Example:

- Project fee: USD 12,000 → USD 15,000
- Support period: 30 days → 60 days
- Payment installments: 4 → 3

7. Preserve exact numbers, dates, names, technical terms,
   and conditions.

8. Do not describe a specific change only in general terms.
   Give the exact old and new values whenever they are available.

9. If something is uncertain or cannot be determined from the
   documents, write:

   "Not clearly determined from the documents."

10. Do not infer that a feature was removed unless the documents
    clearly support that conclusion.

11. If a feature or requirement appears in both documents but is
    described differently, treat it as a difference rather than
    automatically marking it as removed.

12. Separate actual document changes from potential risks.

13. The "risks" section should contain only risks that can
    reasonably result from the actual changes between the documents.

DOCUMENT A:

{document_a}

DOCUMENT B:

{document_b}

Compare the documents and identify:

A. OVERALL COMPARISON

Give a concise description of the overall changes.

B. DIFFERENCES

List all important changed values, dates, amounts, durations,
requirements, conditions, or features.

For numerical or factual changes, use this format:

- Item: OLD VALUE → NEW VALUE

C. ADDED

List information that exists in Document B but not in Document A.

D. REMOVED

List information that exists in Document A but not in Document B.

If nothing was removed, write:

"No information was removed."

E. RISKS

List potential risks caused by the actual changes between
the two documents.

Do not invent risks that are unrelated to the changes.

{format_instructions}

Return ONLY the structured output.
"""


comparison_document_schema = ResponseSchema(
    name="comparison",
    description=(
        "Overall comparison between Document A and Document B."
    )
)

differences_schema = ResponseSchema(
    name="differences",
    description=(
        "Important differences between the two documents."
    )
)

added_schema = ResponseSchema(
    name="added",
    description=(
        "Information present in Document B but not Document A."
    )
)

removed_schema = ResponseSchema(
    name="removed",
    description=(
        "Information present in Document A but not Document B."
    )
)

comparison_risks_schema = ResponseSchema(
    name="risks",
    description=(
        "Potential risks caused by changes between documents."
    )
)


comparison_schemas = [
    comparison_document_schema,
    differences_schema,
    added_schema,
    removed_schema,
    comparison_risks_schema
]


comparison_parser = (
    StructuredOutputParser
    .from_response_schemas(
        comparison_schemas
    )
)


comparison_format_instructions = (
    comparison_parser.get_format_instructions()
)


def prepare_document(pdf_path):

    documents = load_pdf(
        pdf_path
    )

    chunks = text_splitter.split_documents(
        documents
    )

    return {
        "documents": documents,
        "chunks": chunks
    }


def compare_documents(
    pdf_path_a,
    pdf_path_b
):

    document_a = prepare_document(
        pdf_path_a
    )

    document_b = prepare_document(
        pdf_path_b
    )

    document_a_text = "\n\n".join(
        chunk.page_content
        for chunk in document_a["chunks"]
    )

    document_b_text = "\n\n".join(
        chunk.page_content
        for chunk in document_b["chunks"]
    )

    prompt = comparison_template.format(
        document_a=document_a_text,
        document_b=document_b_text,
        format_instructions=(
            comparison_format_instructions
        )
    )

    response = generate_text(
        prompt,
        max_new_tokens=700
    )

    json_text = extract_json_block(
        response
    )

    return comparison_parser.parse(
        json_text
    )


# ============================================================
# 13. STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="AI Document Intelligence",
    page_icon="📄",
    layout="wide"
)


st.title(
    "📄 AI Document Intelligence System"
)


st.write(
    "RAG-based document analysis using "
    "Mistral, FAISS, embeddings, and structured output."
)


# ============================================================
# SYSTEM INFORMATION
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ System Information"
    )

    st.write(
        "**LLM:** Mistral-Nemo-Instruct-2407"
    )

    st.write(
        "**LLM Host:** Kaggle GPU"
    )

    st.write(
        "**API:** Kaggle + FastAPI + ngrok"
    )

    st.write(
        f"**Embedding:** {EMBEDDING_MODEL_NAME}"
    )

    st.divider()

    operation = st.selectbox(
        "Choose Operation",
        [
            "Question Answering",
            "Summarization",
            "Structured Analysis",
            "Document Comparison"
        ]
    )


# ============================================================
# 14. QUESTION ANSWERING
# ============================================================

if operation == "Question Answering":

    st.header(
        "❓ Question Answering"
    )

    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        key="qa_pdf"
    )

    question = st.text_input(
        "Enter your question"
    )

    if st.button("Ask Question"):

        if uploaded_file is None:

            st.error(
                "Please upload a PDF document."
            )

        elif not question.strip():

            st.error(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Analyzing document..."
            ):

                pdf_path = None

                try:

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf"
                    ) as tmp:

                        tmp.write(
                            uploaded_file.getbuffer()
                        )

                        pdf_path = tmp.name

                    documents = load_pdf(
                        pdf_path
                    )

                    chunks = text_splitter.split_documents(
                        documents
                    )

                    vectordb = FAISS.from_documents(
                        chunks,
                        embedding_model
                    )

                    result = answer_question(
                        vectordb,
                        question
                    )

                    st.subheader(
                        "Answer"
                    )

                    st.write(
                        result["answer"]
                    )

                    if result["source_pages"]:

                        st.subheader(
                            "Sources"
                        )

                        st.write(
                            ", ".join(
                                f"Page {p}"
                                for p in result["source_pages"]
                            )
                        )

                except Exception as e:

                    st.error(
                        f"Error: {e}"
                    )

                finally:

                    if (
                        pdf_path
                        and os.path.exists(pdf_path)
                    ):

                        os.unlink(
                            pdf_path
                        )


# ============================================================
# 15. SUMMARIZATION
# ============================================================

elif operation == "Summarization":

    st.header(
        "📝 Document Summarization"
    )

    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        key="summary_pdf"
    )

    if st.button("Summarize Document"):

        if uploaded_file is None:

            st.error(
                "Please upload a PDF document."
            )

        else:

            with st.spinner(
                "Generating summary..."
            ):

                pdf_path = None

                try:

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf"
                    ) as tmp:

                        tmp.write(
                            uploaded_file.getbuffer()
                        )

                        pdf_path = tmp.name

                    result = analyze_document(
                        pdf_path,
                        operation="summary"
                    )

                    st.subheader(
                        "Summary"
                    )

                    st.write(
                        result
                    )

                except Exception as e:

                    st.error(
                        f"Error: {e}"
                    )

                finally:

                    if (
                        pdf_path
                        and os.path.exists(pdf_path)
                    ):

                        os.unlink(
                            pdf_path
                        )


# ============================================================
# 16. STRUCTURED ANALYSIS
# ============================================================

elif operation == "Structured Analysis":

    st.header(
        "📊 Structured Document Analysis"
    )

    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        key="analysis_pdf"
    )

    if st.button("Analyze Document"):

        if uploaded_file is None:

            st.error(
                "Please upload a PDF document."
            )

        else:

            with st.spinner(
                "Analyzing document..."
            ):

                pdf_path = None

                try:

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf"
                    ) as tmp:

                        tmp.write(
                            uploaded_file.getbuffer()
                        )

                        pdf_path = tmp.name

                    result = analyze_document(
                        pdf_path,
                        operation="analysis"
                    )

                    st.subheader(
                        "Document"
                    )

                    st.write(
                        result["document"]
                    )

                    st.subheader(
                        "Summary"
                    )

                    st.write(
                        result["summary"]
                    )

                    st.subheader(
                        "Key Points"
                    )

                    st.write(
                        result["key_points"]
                    )

                    st.subheader(
                        "Risks"
                    )

                    st.write(
                        result["risks"]
                    )

                    st.subheader(
                        "References"
                    )

                    st.write(
                        result["references"]
                    )

                except Exception as e:

                    st.error(
                        f"Error: {e}"
                    )

                finally:

                    if (
                        pdf_path
                        and os.path.exists(pdf_path)
                    ):

                        os.unlink(
                            pdf_path
                        )


# ============================================================
# 17. DOCUMENT COMPARISON
# ============================================================

elif operation == "Document Comparison":

    st.header(
        "🔄 Document Comparison"
    )

    col1, col2 = st.columns(2)

    with col1:

        uploaded_file_a = st.file_uploader(
            "Upload Document A",
            type=["pdf"],
            key="comparison_pdf_a"
        )

    with col2:

        uploaded_file_b = st.file_uploader(
            "Upload Document B",
            type=["pdf"],
            key="comparison_pdf_b"
        )

    if st.button("Compare Documents"):

        if uploaded_file_a is None:

            st.error(
                "Please upload Document A."
            )

        elif uploaded_file_b is None:

            st.error(
                "Please upload Document B."
            )

        else:

            with st.spinner(
                "Comparing documents..."
            ):

                temp_a = None
                temp_b = None

                try:

                    temp_a = tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf"
                    )

                    temp_b = tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf"
                    )

                    temp_a.write(
                        uploaded_file_a.getbuffer()
                    )

                    temp_b.write(
                        uploaded_file_b.getbuffer()
                    )

                    temp_a.close()
                    temp_b.close()

                    result = compare_documents(
                        temp_a.name,
                        temp_b.name
                    )

                    st.subheader(
                        "Overall Comparison"
                    )

                    st.write(
                        result["comparison"]
                    )

                    st.subheader(
                        "🔄 Differences"
                    )

                    st.write(
                        result["differences"]
                    )

                    st.subheader(
                        "➕ Added"
                    )

                    st.write(
                        result["added"]
                    )

                    st.subheader(
                        "➖ Removed"
                    )

                    st.write(
                        result["removed"]
                    )

                    st.subheader(
                        "⚠️ Risks"
                    )

                    st.write(
                        result["risks"]
                    )

                except Exception as e:

                    st.error(
                        f"Error: {e}"
                    )

                finally:

                    if (
                        temp_a
                        and os.path.exists(temp_a.name)
                    ):

                        os.unlink(
                            temp_a.name
                        )

                    if (
                        temp_b
                        and os.path.exists(temp_b.name)
                    ):

                        os.unlink(
                            temp_b.name
                        )