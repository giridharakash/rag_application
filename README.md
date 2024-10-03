# rag_application
RAG Application with Streamlit UI

1. Download and install Python and Ollama

2. To install dependencies run the following command in VS Code terminal-
pip install streamlit numpy langchain faiss-cpu pypdf sentence-transformers

Also, lang-community requires tenacity version <8.4.0 and >8.1.0, Use 
pip install tenacity==8.3.0

Create a venv in your local-model folder 
python -m venv inside your local-model folder

3. To execute the code run the following command in VS Code terminal -
python -m streamlit run RAG_Streamlit_APP.py


Note: This RAG app is built to work only on LLAMA2 library, while using Ollama to install library mention LLAMA2
