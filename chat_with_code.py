import os
from typing import List
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import git

class CodebaseChat:
    def __init__(self, repo_path: str, openai_api_key: str = None):
        """
        Initialize the CodebaseChat with a local repository path
        
        Args:
            repo_path (str): Path to the local git repository
            openai_api_key (str, optional): OpenAI API key. Defaults to None.
        """
        self.repo_path = repo_path
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Initialize components
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separator="\n"
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Load and index the repository
        self.vectorstore = self._index_repository()
        
        # Initialize the conversation chain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
            retriever=self.vectorstore.as_retriever(),
            memory=self.memory,
            return_source_documents=True
        )

    def _get_code_files(self) -> List[str]:
        """Get all code files from the repository."""
        code_files = []
        for root, _, files in os.walk(self.repo_path):
            if ".git" in root:
                continue
            for file in files:
                if file.endswith(('.py', '.js', '.ts', '.tsx', '.jsx', '.java', '.cpp', '.h', '.c', '.hpp', '.cs')):
                    code_files.append(os.path.join(root, file))
        return code_files

    def _index_repository(self):
        """Index the repository content into a vector store."""
        documents = []
        for file_path in self._get_code_files():
            try:
                loader = TextLoader(file_path)
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        texts = self.text_splitter.split_documents(documents)
        return Chroma.from_documents(texts, self.embeddings)

    def ask(self, question: str) -> dict:
        """
        Ask a question about the codebase
        
        Args:
            question (str): The question to ask about the codebase
            
        Returns:
            dict: Contains the answer and source documents
        """
        return self.qa_chain.invoke({"question": question})

def main():
    # Load environment variables
    load_dotenv()
    
    # Get OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY in your environment or .env file")
    
    # Initialize with a sample repository
    repo_path = input("Enter the path to your local git repository: ")
    
    # Create chat instance
    chat = CodebaseChat(repo_path, api_key)
    
    print("\nCodebase Chat initialized! Ask questions about your code (type 'exit' to quit):")
    while True:
        question = input("\nYour question: ")
        if question.lower() == 'exit':
            break
            
        try:
            result = chat.ask(question)
            print("\nAnswer:", result['answer'])
            print("\nSources:")
            for doc in result['source_documents']:
                print(f"- {doc.metadata['source']}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
