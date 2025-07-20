"""
ELC1012 RAG System Configuration File
"""

import os
from typing import Dict, Any

class Config:
    """System configuration class"""
    
    # Document processing configuration
    DOCUMENT_CONFIG = {
        "folder_path": "ELC1012",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "supported_formats": [".pdf"]
    }
    
    # Vector database configuration
    VECTOR_DB_CONFIG = {
        "persist_directory": "./chroma_db",
        "collection_name": "elc1012_documents",
        "embedding_model": "text-embedding-ada-002",
    }
    
    
    # CrewAI agent configuration
    AGENT_CONFIG = {
        "researcher": {
            "role": "ELC1012 Course Research Expert",
            "goal": "Deeply analyze ELC1012 course materials to provide accurate and detailed academic information",
            "backstory": """You are a senior English academic writing expert specializing in ELC1012 course content.
            You are familiar with academic writing standards, paper structure, citation formats, and all aspects.
            You can extract key information from course materials and provide clear, accurate explanations.""",
            "verbose": True,
            "allow_delegation": False,
            "llm_model": "gpt-4o-mini"
        },
        "writer": {
            "role": "Concise Academic Answer Generator",
            "goal": "Generate clear, concise, and well-formatted answers to course-related questions",
            "backstory": """You specialize in presenting academic information in a straightforward and easy-to-read format. 
            You take retrieved data and organize it into brief, structured responses that are easy for students to follow. 
            You avoid unnecessary detail and ensure consistency in layout, using bullet points and simple phrasing.""",
            "verbose": True,
            "allow_delegation": False,
            "llm_model": "gpt-4o-mini"
        }
    }
    
    # CrewAI task configuration
    TASK_CONFIG = {
        "research_task": {
            "description_template": """
            Analyze the user's question: "{user_query}"
            
            Use the RAG tool to retrieve relevant information from ELC1012 course materials.
            Focus on:
            1. Content directly related to the user's question
            2. Specific examples and explanations in course materials
            3. Academic writing standards and guidelines
            
            Please provide detailed analysis results, including:
            - Relevant course content
            - Specific examples and explanations
            - Applicable academic writing principles
            """,
            "expected_output": "Detailed analysis report containing retrieved relevant information and specific examples"
        },
        "writing_task": {
            "description_template": """
            Based on the research agent's analysis results, provide a clear and accurate answer to the user's question: "{user_query}"
            
            Requirements:
            1. Answer should directly address the user's question
            2. Use retrieved course materials as evidence
            3. Provide specific examples and explanations
            4. Maintain academic writing accuracy and professionalism
            5. If information is insufficient, clearly state this
            
            Please organize a structured answer, including:
            - Direct answer to user's question
            - Supporting course content
            - Specific examples or explanations
            - Summary and recommendations
            """,
            "expected_output": "Clear, accurate, and structured answer that directly addresses the user's question"
        }
    }
    
    # RAG tool configuration
    RAG_TOOL_CONFIG = {
        "name": "ELC1012_Knowledge_Base",
        "description": "Retrieve relevant information from ELC1012 course materials"
    }
    
    # System configuration
    SYSTEM_CONFIG = {
        "log_level": "INFO"
    }
    
    @classmethod
    def get_openai_api_key_from_env(cls) -> str:
        """Get OpenAI API key"""
        # First try to get from environment variable
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            return api_key
        return None
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate if configuration is valid"""
        api_key = cls.get_openai_api_key_from_env()
        if not api_key:
            print("Error: OpenAI API key not found")
            print("Please set OPENAI_API_KEY environment variable or configure in .env file")
            return False
        
        # Check if document folder exists
        if not os.path.exists(cls.DOCUMENT_CONFIG["folder_path"]):
            print(f"Error: Document folder {cls.DOCUMENT_CONFIG['folder_path']} does not exist")
            return False
        
        return True
    
    @classmethod
    def get_all_config(cls) -> Dict[str, Any]:
        """Get all configurations"""
        return {
            "document_config": cls.DOCUMENT_CONFIG,
            "vector_db_config": cls.VECTOR_DB_CONFIG,
            "llm_config": cls.LLM_CONFIG,
            "agent_config": cls.AGENT_CONFIG,
            "task_config": cls.TASK_CONFIG,
            "rag_tool_config": cls.RAG_TOOL_CONFIG,
            "system_config": cls.SYSTEM_CONFIG
        } 