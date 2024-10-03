import streamlit as st
import numpy as np
from langchain.llms import Ollama
from langchain.embeddings import LlamaCppEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
import os

# Function to load documents based on category
def load_documents(category):
    data_dir = "data"
    category_map = {
        "Operations": "operations",
        "Academics": "academic",
        "Placement Policies": "placement"
    }
    
    category_folder = category_map.get(category)
    if not category_folder:
        st.error("Invalid category selected.")
        return []
    
    folder_path = os.path.join(data_dir, category_folder)
    if not os.path.exists(folder_path):
        st.error(f"Folder for {category} does not exist.")
        return []
    
    pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pdf')]
    st.write(f"Found {len(pdf_files)} PDF files for category {category}")
    documents = []
    
    for pdf in pdf_files:
        loader = PyPDFLoader(pdf)
        docs = loader.load_and_split()
        documents.extend(docs)

        #debug
    for pdf in pdf_files[:3]:  # Check first 3 PDFs
        loader = PyPDFLoader(pdf)
        docs = loader.load_and_split()
        st.sidebar.write(f"Loaded {len(docs)} pages from {pdf}")
        for i, doc in enumerate(docs[:2]):  # Show first 2 pages of each PDF
            st.sidebar.write(f"Document {i+1} preview: {doc.page_content[:200]}...")
    #debug
    
    return documents

# Function to create or load vector store
def get_vector_store(category):
    index_dir = "vectorstores"
    os.makedirs(index_dir, exist_ok=True)
    index_path = os.path.join(index_dir, f"{category}_faiss.index")
    faiss_store_path = os.path.join(index_dir, f"{category}_faiss.pkl")
    
    if os.path.exists(index_path) and os.path.exists(faiss_store_path):
        # Load existing FAISS index
        vector_store = FAISS.load_local(faiss_store_path, embeddings)
    else:
        # Load documents and create FAISS index
        documents = load_documents(category)
        st.write("Found {len(pdf_files)} PDF files for category {category}")
        if not documents:
            return None
        vector_store = FAISS.from_documents(documents, embeddings)
        vector_store.save_local(faiss_store_path)
    
    return vector_store

# Initialize embeddings
#embeddings = LlamaCppEmbeddings(model_path="C:/Users/chaitanya/.ollama/models")
#embeddings = LlamaCppEmbeddings(model_path="C:/Users/chaitanya/.ollama/models") 
                                #C:/Users/chaitanya/.ollama/models/blobs/sha256-8934d96d3f08982e95922b2b7a2c626a1fe873d7c3b06e8e56d7bc0a1fef9246 # Update with your model path
                                #llama-2-7b.Q4_K_M.gguf

#alternate to the above FAISS LlamaCPPEmbeddings method 

# Custom Embeddings class to integrate SentenceTransformer with FAISS
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts)
    
    # Embeds a single query (e.g., user input)
    def embed_query(self, text):
        return self.model.encode([text])[0]

# Load the SentenceTransformers model
model = SentenceTransformer('all-MiniLM-L6-v2')  # You can choose a different model here

# Function to get vector store using SentenceTransformers embeddings
def get_vector_store(category):
    index_dir = "vectorstores"
    os.makedirs(index_dir, exist_ok=True)
    index_path = os.path.join(index_dir, f"{category}_faiss.index")
    faiss_store_path = os.path.join(index_dir, f"{category}_faiss.pkl")
    st.sidebar.write(f"Building vector store for {category}...")
    
    documents = load_documents(category)  # Ensure this function loads documents properly
    if not documents:
        st.sidebar.error(f"No documents found for {category}")
        return None
    
    st.sidebar.write(f"Loaded {len(documents)} documents for {category}")



    # Create document embeddings
    doc_texts = [doc.page_content for doc in documents]  # Extract text from documents
    embeddings = model.encode(doc_texts)  # Generate embeddings

    # Use the SentenceTransformer embeddings model
    embeddings_model = SentenceTransformerEmbeddings('all-MiniLM-L6-v2')

    # Zip the texts and embeddings together as tuples
    #text_embeddings = list(zip(doc_texts, embeddings))

    # Create a FAISS vector store
    st.sidebar.write("Creating FAISS vector store...")
    vector_store = FAISS.from_texts(doc_texts, embeddings_model)
    # vector_store = FAISS.from_embeddings(text_embeddings)
    # vector_store = FAISS.from_embeddings(embeddings, doc_texts)
    #vector_store = FAISS(embeddings=embeddings, documents=doc_texts)

    st.sidebar.write(f"Saving vector store to {faiss_store_path}")
    vector_store.save_local(faiss_store_path)
    st.sidebar.write(f"Vector store built and saved for {category}")
    return vector_store


# Initialize LLM
llm = Ollama(model="llama2")  # Ensure 'llama2' matches your Ollama model name

# Function to get answer from RAG
def get_answer(category, query):
    vector_store = get_vector_store(category)
    if not vector_store:
        return "No documents found for the selected category."
    
    # Retrieve relevant documents
    docs = vector_store.similarity_search(query, k=5)

    #debug
    st.sidebar.write("Retrieved documents:")
    for i, doc in enumerate(docs):
        st.sidebar.write(f"Doc {i+1} preview: {doc.page_content[:200]}...")
    #debug
    
    # Combine documents into a context
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Create prompt
    #prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    prompt = f"""
    You are an AI assistant for {category} at BITS School of Management. 
    Use only the following context to answer the question. If the answer is not in the context, say "I don't have enough information to answer that question."

    Context:
    {context}

    Question: {query}
    
    Answer:
    """

    
    # Get answer from LLM
    response = llm(prompt)
    st.sidebar.write("LLM Response:")
    st.sidebar.write(response)
    return response

# Streamlit UI
def main():
    st.set_page_config(page_title="🔍 Ask Me Anything - RAG Q&A App", layout="wide", initial_sidebar_state="collapsed")
    st.title("🔍 RAG Q&A Web App")
    
    # Section Divider
    st.markdown("---")
    
    # Sidebar for debug info and vector store rebuilding
    st.sidebar.title("Debug Information")
    
    # Button to rebuild vector stores
    if st.sidebar.button("Rebuild All Vector Stores"):
        progress_placeholder = st.sidebar.empty()
        for i, category in enumerate(["Operations", "Academics", "Placement Policies"]):
            progress_placeholder.info(f"Rebuilding vector store for {category}...")
            get_vector_store(category)
            progress = (i + 1) / 3
            progress_placeholder.progress(progress)
        progress_placeholder.success("All vector stores rebuilt!")

    # Add some custom CSS for styling
    st.markdown(
        """
        <style>
        .stButton button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            padding: 10px 24px;
            border-radius: 8px;
            border: none;
            transition-duration: 0.4s;
        }
        .stButton button:hover {
            background-color: white; 
            color: black; 
            border: 2px solid #4CAF50;
        }
        div.block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

     # Section Divider
    st.markdown("---")

   # col1, col2 = st.columns([2, 1])



     # Dropdown for category selection with centered styling
  #  with col1:
    category = st.selectbox(
        "📂 Select a category:",
        ("Operations", "Academics", "Placement Policies"),
        help="Choose a category to narrow down the source of information."
    )
    
    # Text input for the user's query with a larger input box
    query = st.text_input("💬 Enter your query:", "", help="Ask anything based on the category you selected")
    
    # Add some spacing
    st.markdown(" ")


    #with col2:
    st.markdown("")
    if category == "Operations":
            st.markdown("<b>If you can't find the answer you're looking for, please feel free to reach out to us at <a href='mailto:bitsom.operations@bitsom.edu.in' style='color: blue;'>bitsom.operations@bitsom.edu.in</a> for assistance.</b>", unsafe_allow_html=True)

    if category == "Placement Policies":
        st.markdown("If you can't find the answer you're looking for, please feel free to reach out to us at <a href='mailto:DL-Placement.Committee25@bitsom.edu.in' style='color: blue;'>DL-Placement.Committee25@bitsom.edu.in</a> for assistance.", unsafe_allow_html=True)
    
    if category == "Academics":
        st.markdown("If you can't find the answer you're looking for, please feel free to reach out to us at <a href='mailto:academic.committee22-24@bitsom.edu.in' style='color: blue;'>academic.committee22-24@bitsom.edu.in</a> for assistance.", unsafe_allow_html=True)

    
    
    # Submit button
    if st.button("Get Answer"):
        if not query.strip():
            st.warning("Please enter a query.")
        else:
            with st.spinner("Fetching answer..."):
                answer = get_answer(category, query)
                st.subheader("💡 Here's the answer:")
                st.write(answer)

if __name__ == "__main__":
    main()