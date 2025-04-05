import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from django.conf import settings

def process_pdf(pdf_path, document_id):
    """
    Process a PDF file:
    1. Extract text
    2. Split into chunks
    3. Create embeddings
    4. Store in vector database
    """
    # Load PDF and extract text
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    
    # Create embeddings and store in FAISS vector database
    embeddings = OpenAIEmbeddings()
    
    # Create directory for vector stores if it doesn't exist
    vector_store_dir = os.path.join(settings.BASE_DIR, 'vector_stores')
    os.makedirs(vector_store_dir, exist_ok=True)
    
    # Create a unique path for this document's vector store
    vector_store_path = os.path.join(vector_store_dir, f'document_{document_id}')
    
    # Create and save the vector store
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(vector_store_path)
    
    return vector_store_path

def get_answer(question, vector_store_path):
    """
    Use RAG to answer a question based on the stored PDF content
    """
    # Load the vector store
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local(vector_store_path, embeddings)
    
    # Create a retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    # Create a language model
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    
    # Create a QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    # Get answer
    result = qa_chain({"query": question})
    
    # Extract source page numbers for citation
    source_pages = []
    if 'source_documents' in result:
        for doc in result['source_documents']:
            if 'page' in doc.metadata:
                source_pages.append(doc.metadata['page'])
    
    return {
        'answer': result['result'],
        'source_pages': source_pages
    }
