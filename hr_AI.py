from langchain_ollama import (ChatOllama, OllamaEmbeddings)
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_classic.agents import(tool_calling_agent, AgentExecutor)
from datetime import datetime
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
 
#LLM
llm=ChatOllama(model="qwen3:4b")
 
#Memory
chat_history=InMemoryChatMessageHistory()
 
#Load DB
embeddings=OllamaEmbeddings(model="nomic-embed-text")
 
@tool
def experience_calculator(start_year:str)->str:
    """Calculate candidate Experience"""
    return str(
        datetime.now().year - start_year    
        )
 
@tool
def eligibility_checker(
    skills: str) -> str:
    """Check candidate eligibility"""
 
    required = {
        "python",
        "sql",
        "git"
    }
 
    candidate = {
        skill.strip().lower()
        for skill in skills.split(",")
    }
 
    missing = required - candidate
 
    if len(missing) == 0:
        return "Eligible"
 
    return (
        "Not Eligible. Missing: "
        + ", ".join(missing)
    )
 
@tool
def interview_question(skills:str)->str:
    """Generate 5 interview Questions"""
    prompt=f""" Generate 5 interview question
    for:
    {skills}"""
 
    return llm.invoke(prompt).content

 
 
