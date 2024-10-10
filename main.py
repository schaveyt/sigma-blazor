import os
import glob
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.llms import Anthropic

"""
This script implements a Retrieval-Augmented Generation (RAG) solution for querying
documentation of a C# web framework. It uses a folder of markdown files as its
knowledge base and leverages Anthropic's Claude model for generating responses.

The RAG process involves the following steps:
1. Loading and processing documents
2. Generating embeddings for the documents
3. Storing the embeddings in a vector database
4. Retrieving relevant documents based on a query
5. Using the retrieved documents to generate a response with an LLM

This solution uses various libraries:
- transformers and sentence_transformers for embedding generation
- langchain for document processing, vector storage, and query chain setup
- FAISS for efficient similarity search
- Anthropic's API for accessing the Claude model
"""

class RAGSolution:
    def __init__(self, docs_path: str, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the RAG solution.

        Args:
        docs_path (str): Path to the folder containing markdown files
        model_name (str): Name of the pre-trained model to use for embeddings
        """
        self.docs_path = docs_path
        self.model_name = model_name
        
        # Initialize tokenizer and model for embedding generation
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Initialize the embedding model using HuggingFace's implementation
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        
        # Initialize the text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        # These will be initialized later
        self.vector_store = None
        self.qa_chain = None

    def load_documents(self) -> List[Dict[str, Any]]:
        """
        Load all markdown documents from the specified folder.

        Returns:
        List[Dict[str, Any]]: A list of loaded documents
        """
        documents = []
        # Use glob to find all markdown files in the specified folder
        for file_path in glob.glob(os.path.join(self.docs_path, "*.md")):
            # Use langchain's TextLoader to load each document
            loader = TextLoader(file_path)
            documents.extend(loader.load())
        return documents

    def process_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Process the loaded documents by splitting them into chunks and creating a vector store.

        Args:
        documents (List[Dict[str, Any]]): The list of loaded documents
        """
        # Split the documents into smaller chunks
        texts = self.text_splitter.split_documents(documents)
        # Create a FAISS vector store from the document chunks
        self.vector_store = FAISS.from_documents(texts, self.embeddings)

    def setup_qa_chain(self) -> None:
        """
        Set up the question-answering chain using Anthropic's Claude model and the FAISS vector store.
        """
        # Initialize the Anthropic LLM
        llm = Anthropic(model="claude-2", anthropic_api_key="your_api_key_here")
        
        # Create a retrieval QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # This chain type "stuffs" all retrieved documents into the prompt
            retriever=self.vector_store.as_retriever(),
            return_source_documents=True  # This allows us to see which documents were used to answer
        )

    def query(self, question: str) -> Dict[str, Any]:
        """
        Process a query using the QA chain.

        Args:
        question (str): The query string

        Returns:
        Dict[str, Any]: The response from the QA chain, including the answer and source documents
        """
        if not self.qa_chain:
            raise ValueError("QA chain not set up. Call setup_qa_chain() first.")
        return self.qa_chain({"query": question})

def main():
    """
    Main function to run the RAG solution interactively.
    """
    # Initialize the RAG solution
    rag = RAGSolution("path/to/markdown/files")
    
    # Load and process documents
    documents = rag.load_documents()
    rag.process_documents(documents)
    
    # Set up the QA chain
    rag.setup_qa_chain()

    # Interactive query loop
    while True:
        question = input("Enter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
        result = rag.query(question)
        print(f"Answer: {result['result']}")
        print(f"Source: {result['source_documents'][0].metadata['source']}")

if __name__ == "__main__":
    main()

"""
To use this script:
1. Ensure you have installed all required packages (see requirements.txt)
2. Replace 'path/to/markdown/files' in the main() function with the actual path to your markdown files
3. Replace 'your_api_key_here' in the setup_qa_chain() method with your actual Anthropic API key
4. Run the script and start querying your documentation!

Note: This script assumes that your markdown files are well-structured and contain relevant information.
The quality of the responses depends on the quality and coverage of your documentation.
"""
