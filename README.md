# Monte-Carlo-RAG-Assistant
A Retrieval-Augmented Generation (RAG) system that answers questions about Monte Carlo simulation using a course PDF and open LLMs.
[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Enabled-yellowgreen?logo=langchain)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/Chroma-VectorDB-ff69b4?logo=databricks&logoColor=white)](https://www.trychroma.com/)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Inference-yellow?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io/)

---

## Overview

The application allows users to ask questions about Monte Carlo simulation techniques. It retrieves relevant content from a PDF course document and generates answers using a large language model. The project is built with:

- **LangChain** for document processing and vector search  
- **Chroma** for local vector database storage  
- **Hugging Face Inference API** for multilingual embeddings and LLM inference  
- **Streamlit** for the interactive front-end interface

## Features

- Parses and splits a PDF file into content chunks  
- Generates embeddings using a multilingual model (`intfloat/multilingual-e5-large`)  
- Stores chunks in a Chroma vector store  
- Retrieves the top relevant chunks using similarity search  
- Generates plain-text answers using `mistralai/Mixtral-8x7B-Instruct-v0.1`  
- Provides clear mathematical explanations without LaTeX or Markdown formatting

## Project Structure
\begin{verbatim}
├── cm_monte_carlo.pdf    # Source PDF document used for retrieval
├── database.py           # Embedding and indexing pipeline
├── interface.py          # Streamlit interface for querying
├── chroma_db/            # Persisted Chroma vector database
\end{verbatim}


## Example Questions

You can ask:

- What is the Law of Large Numbers?  
- How does stratified sampling reduce variance?  
- What is the rejection sampling method?

---

## Notes

- All answers are based solely on the content of the course PDF.  
- If no relevant chunk is found, the app will return a fallback message.  
- Mathematical expressions are converted to plain-text ASCII notation for compatibility.

---

## Acknowledgments

- Course material from Université Paris Dauphine | PSL  
- Hugging Face for access to embedding and instruction-tuned models

---

