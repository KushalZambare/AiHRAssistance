import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
 
#LLM
embedding=OllamaEmbeddings(model="nomic-embed-text")
 
documents=[]
for file in os.listdir("documents"):
    path=os.path.join("documents",file)
    loader=TextLoader(path)
    docs=loader.load()
    documents.extend(docs)
print(f"Loaded{len(documents)} documents")
 
 
#Split Document
 
splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
 
chunks=splitter.split_documents(
    documents
)
 
print(f"Created {len(chunks)} chunks")
 
vector_db=FAISS.from_documents(chunks,embedding)
 
vector_db.save_local(
    "HrVectorDB"
)
 
print("Database Created")
 