# ELC1012 RAG System

This is a RAG (Retrieval-Augmented Generation) system based on the CrewAI framework, specifically designed to process ELC1012 course materials. The system can intelligently retrieve and analyze course content, providing users with accurate and detailed academic information.

## Features

- **Intelligent Document Processing**: Automatically scans all PDF documents in the ELC1012 folder, supporting recursive processing of subfolders
- **Vectorized Storage**: Uses OpenAI's text-embedding-ada-002 model and Chroma vector database to store document embeddings
- **Multi-Agent Collaboration**: Uses CrewAI's research agent and writing agent to collaboratively process user queries
- **Intelligent Retrieval**: Retrieves relevant course content based on semantic similarity
- **Structured Responses**: Provides clear, accurate, and structured answers
- **Persistent Storage**: Supports vector database persistence to avoid repeated processing
- **Document Update Detection**: Automatically detects document updates and rebuilds the vector database

## System Architecture

### Main Components

1. **Document Loader**: Uses PyPDFLoader to recursively scan the ELC1012 folder and load all PDF documents
2. **Text Splitter**: Uses RecursiveCharacterTextSplitter to split documents into chunks suitable for vectorization
3. **Vector Database**: Uses Chroma to store document embeddings with persistence support
4. **RAG Tool**: Custom RAG tool for retrieving relevant information
5. **Agent System**: 
   - Research Agent: Responsible for retrieving and analyzing information, using gpt-4o-mini model
   - Writing Agent: Responsible for organizing and presenting information, using gpt-4o-mini model

### Workflow

1. User inputs a query
2. Research agent uses RAG tool to retrieve relevant course content
3. Research agent analyzes the retrieved information
4. Writing agent generates structured answers based on analysis results
5. Returns final answer to user

## Installation and Setup

### Requirements

- Python 3.8+
- OpenAI API key

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure OpenAI API Key

There are two ways to configure the API key:

**Method 1: Environment Variable**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

**Method 2: main.py file**
Create a .main.py file and add:
```
OPENAI_API_KEY=your-api-key-here
```

### Ensure Document Structure

Ensure the ELC1012 folder contains all course material PDF files. The document structure during system development:

```
ELC1012/
├── ELC1012_1013.pdf
├── Assessment 2 Guide.pdf
├── Assessment 3 Guide.pdf
├── lesson1/
├── lesson2/
├── lesson3/
├── lesson4/
├── lesson5/
├── lesson6/
├── lesson7/
├── lesson8/
├── lesson10/
└── lesson11/
```

## Usage

### Run the System

```bash
python main.py
```

### Usage Process

1. After starting the program, the system will automatically load all PDF documents in the ELC1012 folder
2. The system will create a vector database (first run may take some time)
3. Once the system is ready, you can start asking questions
4. Enter your question, and the system will provide answers based on course materials
5. Enter 'quit' or 'exit' to exit the program

### Example Questions

- "What is the citation format for academic writing?"
- "How to write an effective paper introduction?"
- "What are the requirements for Assessment 2?"
- "How to avoid plagiarism in academic writing?"
- "What are the main contents of the ELC1012 course?"

### Programming Interface Usage

```python
from main import ELC1012RAGSystem

# Initialize system
rag_system = ELC1012RAGSystem()
rag_system.initialize_system()

# Process query
result = rag_system.process_query("What is academic writing?")
print(result)
```

## Technical Details

### Document Processing

- Uses PyPDFLoader to load PDF documents
- Uses RecursiveCharacterTextSplitter for text splitting
- Chunk size: 1000 characters, overlap: 200 characters
- Supports recursive scanning of subfolders

### Vectorization

- Uses OpenAI's text-embedding-ada-002 model
- Uses Chroma as vector database
- Persistent storage to ./chroma_db directory
- Automatically detects document updates and rebuilds database

### Model Configuration

The system uses the following model configuration:

- **Research Agent**: gpt-4o-mini
- **Writing Agent**: gpt-4o-mini
- **Embedding Model**: text-embedding-ada-002

### Agent Configuration

- **Research Agent**: Focuses on retrieving and analyzing course content, providing detailed analysis reports
- **Writing Agent**: Focuses on organizing and presenting information, generating structured answers
- Uses sequential processing mode to ensure consistency of information flow

## Configuration

System configuration is in the `config.py` file, main configuration items include:

### Document Processing Configuration
```python
DOCUMENT_CONFIG = {
    "folder_path": "ELC1012",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "supported_formats": [".pdf"]
}
```

### Vector Database Configuration
```python
VECTOR_DB_CONFIG = {
    "persist_directory": "./chroma_db",
    "collection_name": "elc1012_documents",
    "embedding_model": "text-embedding-ada-002",
}
```

### Agent Configuration
```python
AGENT_CONFIG = {
    "researcher": {
        "role": "ELC1012 Course Research Expert",
        "goal": "Deeply analyze ELC1012 course materials...",
        "llm_model": "gpt-4o-mini"
    },
    "writer": {
        "role": "Academic Writing Assistant", 
        "goal": "Organize retrieved information...",
        "llm_model": "gpt-4o-mini"
    }
}
```

## Project Structure

```
ENG_RAG/
├── main.py                 # Main program file
├── config.py               # Configuration file
├── requirements.txt        # Dependency package list
├── README.md              # Project description
├── PROJECT_SUMMARY.md     # Project summary
├── chroma_db/             # Vector database storage directory
└── ELC1012/              # Course materials folder
    ├── lesson1/
    ├── lesson2/
    ├── ...
    ├── ELC1012_1013.pdf
    ├── Assessment 2 Guide.pdf
    └── Assessment 3 Guide.pdf
``` 