import os
import config
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

class PDFRAG:
    def __init__(self, data_dir: str, google_api_key: str):
        self.data_dir = data_dir
        os.environ["GOOGLE_API_KEY"] = google_api_key
        self.documents = self.load_documents()
        self.vector_store = self.create_vector_store()
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

    def create_vector_store(self):
        print("Creating vector store...")
        text_splitter = \
        RecursiveCharacterTextSplitter(chunk_size=1000,\
                                       chunk_overlap=200)
        texts = text_splitter.split_documents(self.documents)
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            task_type="retrieval_document",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        return FAISS.from_documents(texts, embeddings)

    def setup_qa_chain(self):
        print("Setting up QA chain...")
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Define the prompt template
        template = """Use the following pieces of \
            context to answer the question at the \
            end. If you don't know the answer, \
            just say that you don't know, don't \
                try to make up an answer.
        
        Context: {context}
        Question: {question}
        
        Answer:"""
        
        # Create the retriever
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        self.qa_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | PromptTemplate.from_template(template)
            | llm
        )
        
        return self.qa_chain

    def query(self, question: str) -> tuple:
        print("Processing query...")
        # Get documents for the question
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        docs = retriever.get_relevant_documents(question)
        
        # Get the answer
        answer = self.qa_chain.invoke(question)
        
        return answer.content, docs

    def display_relevant_docs(self, docs, num_docs=3):
        print(f"\nTop {num_docs} most relevant document chunks:")
        for i, doc in enumerate(docs[:num_docs], 1):
            print(f"\nDocument {i}:")
            print(f"Content: {doc.page_content[:200]}...")
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")

def main():
    data_dir = "data"
    google_api_key = config.GOOGLE_API_KEY
    
    rag = PDFRAG(data_dir, google_api_key)
    
    print("Ready to answer questions. Type 'quit' to exit.")
    while True:
        user_question = input("\nEnter your question: ")
        if user_question.lower() == 'quit':
            break
        answer, source_docs = rag.query(user_question)
        print("\nAnswer:", answer)
        rag.display_relevant_docs(source_docs)
        print("\n" + "-"*50)

    print("Thank you for using the PDF RAG system!")

if __name__ == "__main__":
    main()