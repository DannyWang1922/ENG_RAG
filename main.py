import os
import glob
import logging
from typing import List
from pathlib import Path

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
from pydantic import Field

from config import Config

import os
os.environ["CHROMA_TELEMETRY_ENABLED"] = "False"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main.py')

class RAGTool(BaseTool):
    """Custom RAG Tool"""
    vector_store: Chroma = Field(description="Vector database")
    
    def __init__(self, vector_store):
        super().__init__(
            name=Config.RAG_TOOL_CONFIG["name"],
            description=Config.RAG_TOOL_CONFIG["description"],
            vector_store=vector_store
        )
    
    def _run(self, query: str) -> str:
        """Execute RAG search"""
        try:
            # Use vector database for similarity search
            docs = self.vector_store.similarity_search(query, k=5)
            
            if not docs:
                return "No relevant information found."
            
            # Format search results
            result = "Found the following relevant information:\n\n"
            for i, doc in enumerate(docs, 1):
                result += f"{i}. Source: {doc.metadata.get('filename', 'Unknown file')}\n"
                result += f"   Content: {doc.page_content}\n\n"
            
            return result
            
        except Exception as e:
            return f"Error occurred during search: {str(e)}"

class ELC1012RAGSystem:
    def __init__(self, openai_api_key: str = None):
        """
        Initialize ELC1012 RAG system
        
        Args:
            openai_api_key: OpenAI API key, if None will get from config file
        """
        self.openai_api_key = openai_api_key or Config.get_openai_api_key_from_env()
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found")
            
        # Load system configuration
        self.log_level = Config.SYSTEM_CONFIG["log_level"]
            
        self.vector_store = None
        self.rag_tool = None
        self.setup_environment()
    
    def setup_environment(self):
        """Set up environment variables"""
        os.environ["OPENAI_API_KEY"] = self.openai_api_key
        
        # Update logging level if different from default
        if self.log_level != "INFO":
            logging.getLogger().setLevel(getattr(logging, self.log_level))

    def initialize_system(self):
        """Initialize system"""
        logger.info("Checking for existing vector database...")
        
        # Check if vector database already exists
        if self.check_vector_db_exists():
            logger.info("Found existing vector database, checking for updates...")
            
            # Check if documents have been updated
            if self.check_documents_updated():
                logger.info("Documents have been updated, recreating vector database...")
                self.force_recreate_vector_db()
                return
            else:
                logger.info("No document updates detected, loading existing database...")
                if self.load_existing_vector_store():
                    logger.info("Successfully loaded existing vector database!")
                    logger.info("Setting up RAG tool...")
                    self.setup_rag_tool()
                    logger.info("System initialization completed!")
                    return
                else:
                    logger.info("Failed to load existing database, will create new one...")
        
        # If no existing database or failed to load, create new one
        logger.info("Loading ELC1012 course materials...")
        documents = self.load_documents()
        
        if not documents:
            raise ValueError("No PDF documents found")
            
        logger.info("Creating vector database...")
        self.create_vector_store(documents)
        
        logger.info("Setting up RAG tool...")
        self.setup_rag_tool()
        
        logger.info("System initialization completed!")

    def check_vector_db_exists(self) -> bool:
        """
        Check if vector database already exists
        
        Returns:
            bool: True if database exists, False otherwise
        """
        persist_dir = Config.VECTOR_DB_CONFIG["persist_directory"]
        collection_name = Config.VECTOR_DB_CONFIG["collection_name"]
        
        # Check if persist directory exists and has content
        if not os.path.exists(persist_dir):
            return False
            
        # Check if chroma database files exist
        chroma_files = ["chroma.sqlite3"]
        for file in chroma_files:
            if not os.path.exists(os.path.join(persist_dir, file)):
                return False
                
        return True
    
    def load_existing_vector_store(self):
        """
        Load existing vector database
        
        Returns:
            bool: True if successfully loaded, False otherwise
        """
        try:
            embeddings = OpenAIEmbeddings(model=Config.VECTOR_DB_CONFIG["embedding_model"])
            self.vector_store = Chroma(
                persist_directory=Config.VECTOR_DB_CONFIG["persist_directory"],
                embedding_function=embeddings,
                collection_name=Config.VECTOR_DB_CONFIG["collection_name"]
            )
            
            # Test if the collection has data
            collection = self.vector_store._collection
            count = collection.count()
            
            if count > 0:
                logger.info(f"Successfully loaded existing vector database with {count} documents")
                logger.info(f"Loaded existing vector database with {count} documents")
                return True
            else:
                logger.info("Vector database exists but is empty")
                logger.warning("Vector database exists but is empty")
                return False
                
        except Exception as e:
            logger.info(f"Failed to load existing vector database: {str(e)}")
            logger.error(f"Failed to load existing vector database: {str(e)}")
            return False

    def force_recreate_vector_db(self):
        """Force recreation of vector database"""
        logger.info("Force recreating vector database...")
        
        # Remove existing database files
        persist_dir = Config.VECTOR_DB_CONFIG["persist_directory"]
        if os.path.exists(persist_dir):
            import shutil
            shutil.rmtree(persist_dir)
            logger.info(f"Removed existing database directory: {persist_dir}")
        
        # Recreate database directly
        logger.info("Loading ELC1012 course materials...")
        documents = self.load_documents()
        
        if not documents:
            raise ValueError("No PDF documents found")
            
        logger.info("Creating vector database...")
        self.create_vector_store(documents)
        
        logger.info("Setting up RAG tool...")
        self.setup_rag_tool()
        
        logger.info("System initialization completed!")

    def check_documents_updated(self) -> bool:
        """
        Check if documents have been updated since last database creation
        
        Returns:
            bool: True if documents have been updated, False otherwise
        """
        try:
            # Get list of current documents
            folder_path = Config.DOCUMENT_CONFIG["folder_path"]
            supported_formats = Config.DOCUMENT_CONFIG["supported_formats"]
            
            current_files = []
            for format_ext in supported_formats:
                files = glob.glob(f"{folder_path}/**/*{format_ext}", recursive=True)
                current_files.extend(files)
            
            # Get file modification times
            current_file_info = {}
            for file in current_files:
                mtime = os.path.getmtime(file)
                current_file_info[file] = mtime
            
            # Check if we have stored file info
            db_info_file = os.path.join(Config.VECTOR_DB_CONFIG["persist_directory"], "file_info.json")
            
            if not os.path.exists(db_info_file):
                return True  # No previous info, assume updated
            
            import json
            with open(db_info_file, 'r') as f:
                stored_file_info = json.load(f)
            
            # Compare current files with stored files
            if set(current_file_info.keys()) != set(stored_file_info.keys()):
                return True  # File list changed
            
            # Check if any files have been modified
            for file, current_mtime in current_file_info.items():
                if file not in stored_file_info or stored_file_info[file] != current_mtime:
                    return True  # File modified
            
            return False  # No changes detected
            
        except Exception as e:
            logging.warning(f"Error checking document updates: {str(e)}")
            return True  # Assume updated if error
        

        
    def load_documents(self, folder_path: str = None) -> List[Document]:
        """
        Load all PDF documents from ELC1012 folder
        
        Args:
            folder_path: Document folder path, if None will use path from config file
            
        Returns:
            List[Document]: List of loaded documents
        """
        folder_path = folder_path or Config.DOCUMENT_CONFIG["folder_path"]
        documents = []
        
        # Use supported formats from config
        supported_formats = Config.DOCUMENT_CONFIG["supported_formats"]
        
        logger.info(f"Loading documents from {folder_path} with supported formats: {supported_formats}")
        
        # Recursively find all supported files
        all_files = []
        for format_ext in supported_formats:
            files = glob.glob(f"{folder_path}/**/*{format_ext}", recursive=True)
            all_files.extend(files)
        
        logger.info(f"Found {len(all_files)} supported files:")
        for file in all_files:
            logger.info(f"  - {file}")
            
        for file in all_files:
            try:
                loader = PyPDFLoader(file)
                docs = loader.load()
                
                # Add metadata for each document
                for doc in docs:
                    doc.metadata["source"] = file
                    doc.metadata["filename"] = os.path.basename(file)
                    
                documents.extend(docs)
                logger.info(f"Successfully loaded: {file} ({len(docs)} pages)")
                logger.info(f"Successfully loaded {file} with {len(docs)} pages")
                
            except Exception as e:
                logger.info(f"Failed to load {file}: {str(e)}")
                logger.error(f"Failed to load {file}: {str(e)}")
                
        return documents
    
    def create_vector_store(self, documents: List[Document]):
        """
        Create vector database
        
        Args:
            documents: Document list
        """
        # Text splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.DOCUMENT_CONFIG["chunk_size"],
            chunk_overlap=Config.DOCUMENT_CONFIG["chunk_overlap"],
            length_function=len,
        )
        
        splits = text_splitter.split_documents(documents)
        logger.info(f"Documents split into {len(splits)} chunks")
        
        # Create vector database
        embeddings = OpenAIEmbeddings(model=Config.VECTOR_DB_CONFIG["embedding_model"])
        self.vector_store = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=Config.VECTOR_DB_CONFIG["persist_directory"],
            collection_name=Config.VECTOR_DB_CONFIG["collection_name"],
        )
        
        # Save file information for future update checks
        self._save_file_info()
        
        logger.info("Vector database creation completed")
    
    def _save_file_info(self):
        """Save current file information for update checking"""
        try:
            folder_path = Config.DOCUMENT_CONFIG["folder_path"]
            supported_formats = Config.DOCUMENT_CONFIG["supported_formats"]
            
            current_files = []
            for format_ext in supported_formats:
                files = glob.glob(f"{folder_path}/**/*{format_ext}", recursive=True)
                current_files.extend(files)
            
            # Get file modification times
            file_info = {}
            for file in current_files:
                mtime = os.path.getmtime(file)
                file_info[file] = mtime
            
            # Save to file
            import json
            db_info_file = os.path.join(Config.VECTOR_DB_CONFIG["persist_directory"], "file_info.json")
            with open(db_info_file, 'w') as f:
                json.dump(file_info, f, indent=2)
                
            logging.info(f"Saved file info for {len(file_info)} files")
            
        except Exception as e:
            logger.error(f"Failed to save file info: {str(e)}")
    
    def setup_rag_tool(self):
        """Set up RAG tool"""
        self.rag_tool = RAGTool(self.vector_store)
        
    def create_agents(self):
        """Create CrewAI agents"""
        
        # Research agent - responsible for retrieving and analyzing information
        researcher_config = Config.AGENT_CONFIG["researcher"]
        researcher_llm = ChatOpenAI(
            model=researcher_config["llm_model"],
        )
        researcher = Agent(
            role=researcher_config["role"],
            goal=researcher_config["goal"],
            backstory=researcher_config["backstory"],
            tools=[self.rag_tool],
            verbose=researcher_config["verbose"],
            allow_delegation=researcher_config["allow_delegation"],
            llm=researcher_llm
        )
        
        # Writer agent - responsible for organizing and presenting information
        writer_config = Config.AGENT_CONFIG["writer"]
        writer_llm = ChatOpenAI(
            model=writer_config["llm_model"],
        )
        writer = Agent(
            role=writer_config["role"],
            goal=writer_config["goal"],
            backstory=writer_config["backstory"],
            verbose=writer_config["verbose"],
            allow_delegation=writer_config["allow_delegation"],
            llm=writer_llm
        )
        
        return researcher, writer
    
    def create_tasks(self, researcher, writer, user_query: str):
        """Create tasks"""
        
        # Research task
        research_task_config = Config.TASK_CONFIG["research_task"]
        research_task = Task(
            description=research_task_config["description_template"].format(user_query=user_query),
            agent=researcher,
            expected_output=research_task_config["expected_output"]
        )
        
        # Writing task
        writing_task_config = Config.TASK_CONFIG["writing_task"]
        writing_task = Task(
            description=writing_task_config["description_template"].format(user_query=user_query),
            agent=writer,
            expected_output=writing_task_config["expected_output"]
        )
        
        return research_task, writing_task
    
    def process_query(self, user_query: str) -> str:
        """
        Process user query
        
        Args:
            user_query: User's question
            
        Returns:
            str: System response
        """
        # Create agents
        researcher, writer = self.create_agents()
        
        # Create tasks
        research_task, writing_task = self.create_tasks(researcher, writer, user_query)
        
        # Create Crew
        crew = Crew(
            agents=[researcher, writer],
            tasks=[research_task, writing_task],
            process=Process.sequential,
            verbose=True
        )
        
        # Execute tasks
        result = crew.kickoff()
        
        return result
    

def main():
    """Main function"""
    print("=== ELC1012 RAG System ===")
    
    # Validate configuration
    if not Config.validate_config():
        return
    
    try:
        # Initialize system
        rag_system = ELC1012RAGSystem(openai_api_key)
        rag_system.initialize_system()
        
        logger.info("System is ready! You can start asking questions.")
        print()
        print("Enter 'quit' or 'exit' to exit the program.")
        
        # Interactive loop
        while True:
            user_query = input("\nPlease enter your question (OR 'quit' or 'exit' to exit): ").strip()
            
            if user_query.lower() in ['quit', 'exit']:
                print("Thank you for using the ELC1012 RAG system!")
                break
                
            if not user_query:
                print("Please enter a valid question.")
                continue
                
            try:
                print("\nProcessing your question...")
                result = rag_system.process_query(user_query)
                print(f"\nAnswer:\n{result}")
                
            except Exception as e:
                logger.info(f"Error occurred while processing question: {str(e)}")
                
    except Exception as e:
        logger.info(f"System initialization failed: {str(e)}")

if __name__ == "__main__":
    openai_api_key = None # your-api-key-here
    main(openai_api_key)
