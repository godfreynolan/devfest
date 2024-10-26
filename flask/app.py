from flask import Flask, request, render_template
import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
import config

app = Flask(__name__)

class PDFRAG:
    def __init__(self, data_dir: str, google_api_key: str, index_path: str = "faiss_index"):
        self.data_dir = data_dir
        self.index_path = index_path
        os.environ["GOOGLE_API_KEY"] = google_api_key
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            task_type="retrieval_document",
            google_api_key=google_api_key
        )
        self.vector_store = self.load_or_create_vector_store()
        self.qa_chain = self.setup_qa_chain()

    def load_documents(self) -> List:
        print("Loading PDF files...")
        documents = []
        for file in os.listdir(self.data_dir):
            if file.endswith(".pdf"):
                file_path = os.path.join(self.data_dir, file)
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
        return documents

    def load_or_create_vector_store(self):
        # Try to load existing index
        if os.path.exists(self.index_path):
            print("Loading existing vector store...")
            return FAISS.load_local(
                self.index_path,
                self.embeddings
            )
        
        # Create new index if none exists
        print("Creating new vector store...")
        documents = self.load_documents()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)
        
        vector_store = FAISS.from_documents(texts, self.embeddings)
        
        # Save the index
        print("Saving vector store...")
        vector_store.save_local(self.index_path)
        
        return vector_store

    def update_vector_store(self):
        """Method to update the vector store with new documents"""
        documents = self.load_documents()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)
        
        # Add new documents to existing index
        self.vector_store.add_documents(texts)
        
        # Save updated index
        self.vector_store.save_local(self.index_path)
        print("Vector store updated and saved.")

    def setup_qa_chain(self):
        print("Setting up QA chain...")
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context: {context}
        Question: {question}
        
        Answer:"""
        
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        qa_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | PromptTemplate.from_template(template)
            | llm
        )
        
        return qa_chain

    def query(self, question: str) -> tuple:
        print("Processing query...")
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        docs = retriever.get_relevant_documents(question)
        answer = self.qa_chain.invoke(question)
        return answer.content, docs

    def display_relevant_docs(self, docs, num_docs=3):
        print(f"\nTop {num_docs} most relevant document chunks:")
        for i, doc in enumerate(docs[:num_docs], 1):
            print(f"\nDocument {i}:")
            print(f"Content: {doc.page_content[:200]}...")
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")


# Create a global RAG instance
data_dir = "data"
google_api_key = config.GOOGLE_API_KEY
rag_instance = None

def get_rag_instance():
    global rag_instance
    if rag_instance is None:
        rag_instance = PDFRAG(data_dir, google_api_key)
    return rag_instance

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    rag = get_rag_instance()
    user_question = request.args.get('msg')
    answer, source_docs = rag.query(user_question)
    return answer

@app.route("/update_index")
def update_index():
    rag = get_rag_instance()
    rag.update_vector_store()
    return "Vector store updated successfully!"

if __name__ == "__main__":
    app.run()
