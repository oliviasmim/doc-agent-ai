"""
JavaScript Code Review RAG System

This script uses LangChain and OpenAI to create a Retrieval Augmented Generation (RAG)
system for JavaScript code review. It loads JavaScript files from a repository, creates embeddings,
and uses a retrieval chain to provide code review suggestions.
"""

import os
import glob
from git import Repo
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import LangChain components
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


def clone_repository(repo_url, repo_path):
    """Clone a repository if it doesn't exist already"""
    if not os.path.exists(repo_path):
        print(f"Cloning repository {repo_url} to {repo_path}...")
        Repo.clone_from(repo_url, to_path=repo_path)
        print("Repository cloned successfully.")
    else:
        print(f"Repository already exists at {repo_path}")


def load_code_documents(repo_path, subfolder):
    """Load JavaScript code documents from the repository"""
    documents = []
    base_path = os.path.join(repo_path, subfolder)
    extensions = [".js", ".jsx", ".ts", ".tsx"]
    exclude_patterns = ["node_modules", "dist", "build"]

    print("Loading documents...")
    for root, dirs, files in os.walk(base_path):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_patterns]
        
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                try:
                    loader = TextLoader(file_path)
                    documents.extend(loader.load())
                    print(f"Loaded {file_path}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

    print(f"Loaded {len(documents)} documents.")
    return documents


def split_documents(documents):
    """Split documents into chunks"""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=50
    )

    print("Splitting documents...")
    texts = text_splitter.split_documents(documents)
    print(f"Created {len(texts)} text chunks.")
    return texts


def create_vector_store(texts):
    """Create a vector store from the text chunks"""
    print("Creating vector store...")
    db = Chroma.from_documents(texts, OpenAIEmbeddings(disallowed_special=()))
    return db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8},
    )


def setup_retrieval_chain():
    """Set up the retrieval chain for code review"""
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        max_tokens=400,  # Increased for more detailed responses
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an experienced JavaScript/TypeScript code reviewer. Provide detailed information about the code review and suggest improvements based on JavaScript/React best practices, considering the context provided below: \n\n{context}",
        ),
        ("user", "{input}"),
    ])

    document_chain = create_stuff_documents_chain(llm, prompt)
    return document_chain


def main():
    # Define repository path
    repo_path = "./test_repo"
    repo_url = "https://github.com/oliviasmim/typescript-react-dashboard.git"
    subfolder = "frontend/src"
    
    # Clone repository
    clone_repository(repo_url, repo_path)
    
    # Load documents
    documents = load_code_documents(repo_path, subfolder)
    
    # Split documents
    texts = split_documents(documents)
    
    # Create vector store
    retriever = create_vector_store(texts)
    
    # Set up retrieval chain
    document_chain = setup_retrieval_chain()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    # Example code review
    print("\nPerforming code review...")
    response = retrieval_chain.invoke({
        "input": "Create an overview of the codebase and identify any React best practices that could be improved.",
    })
    
    print("\nCode Review Result:")
    print(response["answer"])


if __name__ == "__main__":
    main()