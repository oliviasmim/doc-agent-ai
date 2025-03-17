"""
Code Review RAG System

This script uses LangChain and OpenAI to create a Retrieval Augmented Generation (RAG)
system for code review. It loads Python files from a repository, creates embeddings,
and uses a retrieval chain to provide code review suggestions.
"""

import os
from git import Repo
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import LangChain components
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
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
    """Load code documents from the repository"""
    loader = GenericLoader.from_filesystem(
        os.path.join(repo_path, subfolder),
        glob="**/*",
        suffixes=[".py"],
        exclude=["**/non-utf-8-encoding.py"],
        parser=LanguageParser(language=Language.PYTHON, parser_threshold=500)
    )

    print("Loading documents...")
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")
    return documents


def split_documents(documents):
    """Split documents into chunks"""
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
    )

    print("Splitting documents...")
    texts = python_splitter.split_documents(documents)
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
        max_tokens=200,
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Você é um revisor de código experiente. Forneça informações detalhadas sobre a revisão do código e sugestões de melhorias baseadas no contexto fornecido abaixo: \n\n{context}",
        ),
        ("user", "{input}"),
    ])

    document_chain = create_stuff_documents_chain(llm, prompt)
    return document_chain


def main():
    # Define repository path
    repo_path = "./test_repo"
    repo_url = "https://github.com/langchain-ai/langchain"
    subfolder = "libs/core/langchain_core/"
    
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
        "input": "Você pode revisar e sugerir melhorias para o código de RunnableBinding"
    })
    
    print("\nCode Review Result:")
    print(response["answer"])


if __name__ == "__main__":
    main()