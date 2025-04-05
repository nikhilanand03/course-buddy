import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
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
    
    # Create embeddings using Azure OpenAI
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-ada-002",  # Use your embedding deployment name
        api_version="2024-02-01",  # Update to your API version
    )
    
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
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-3-small",  # Use your embedding deployment name
        api_version="2024-02-01",  # Update to your API version
    )
    vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
    
    # Create a retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    # Create a language model using Azure OpenAI with GPT-4o
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4o",  # Your GPT-4o deployment name
        api_version="2024-02-01",  # Your API version
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    
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