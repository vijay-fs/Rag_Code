import os  # Importing the os module to interact with the operating system
from typing import List  # Importing List for type hinting
from dotenv import load_dotenv  # Importing load_dotenv to load environment variables from a .env file
from langchain_community.document_loaders import TextLoader  # Importing TextLoader to read text from files
from langchain.text_splitter import CharacterTextSplitter  # Importing CharacterTextSplitter to split text into chunks
from langchain_community.vectorstores import Chroma  # Importing Chroma for vector-based document storage
from langchain_openai import OpenAIEmbeddings  # Importing OpenAIEmbeddings to convert text into embeddings
from langchain_openai import ChatOpenAI  # Importing ChatOpenAI to interact with OpenAI's language model
from langchain.chains import ConversationalRetrievalChain  # Importing ConversationalRetrievalChain for retrieval-based conversations
from langchain.memory import ConversationBufferMemory  # Importing ConversationBufferMemory to store conversation history
import git  # Importing git module to interact with Git repositories

class CodebaseChat:
    def __init__(self, repo_path: str, openai_api_key: str = None):
        """
        Initialize the CodebaseChat with a local repository path
        
        Args:
            repo_path (str): Path to the local git repository
            openai_api_key (str, optional): OpenAI API key. Defaults to None.
        """
        self.repo_path = repo_path  # Store the repository path
        
        # Set OpenAI API key as an environment variable if provided
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Initialize OpenAI embeddings for text processing
        self.embeddings = OpenAIEmbeddings()
        
        # Initialize a text splitter to break documents into smaller chunks
        self.text_splitter = CharacterTextSplitter(
            chunk_size=1000,  # Define chunk size of 1000 characters
            chunk_overlap=200,  # Overlap of 200 characters between chunks to preserve context
            separator="\n"  # Split based on newline characters
        )
        
        # Initialize memory to store conversation history
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",  # Key for storing chat history
            return_messages=True,  # Return previous messages in the chat
            output_key="answer"  # Store output under the 'answer' key
        )
        
        # Index the repository content into a vector store
        self.vectorstore = self._index_repository()
        
        # Create a conversational retrieval chain using an OpenAI model
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),  # Use GPT-3.5 Turbo model with zero temperature (deterministic responses)
            retriever=self.vectorstore.as_retriever(),  # Use the vector store as a retriever
            memory=self.memory,  # Attach conversation memory
            return_source_documents=True  # Return source documents for reference
        )

    def _get_code_files(self) -> List[str]:
        """Get all code files from the repository."""
        code_files = []  # Initialize an empty list to store file paths
        
        # Walk through the repository directory
        for root, _, files in os.walk(self.repo_path):
            if ".git" in root:  # Skip Git metadata folders
                continue
            
            # Filter and collect only specific programming language files
            for file in files:
                if file.endswith((
                    '.py', '.js', '.ts', '.tsx', '.jsx', '.java', '.cpp', '.h', '.c', '.hpp', '.cs'
                )):
                    code_files.append(os.path.join(root, file))  # Append the full file path
        
        return code_files  # Return the list of code file paths

    def _index_repository(self):
        """Index the repository content into a vector store."""
        documents = []  # Initialize an empty list to store loaded documents
        
        # Iterate through all code files in the repository
        for file_path in self._get_code_files():
            try:
                loader = TextLoader(file_path)  # Load file content using TextLoader
                documents.extend(loader.load())  # Append loaded content to the document list
            except Exception as e:
                print(f"Error loading {file_path}: {e}")  # Print an error message if loading fails

        # Split documents into smaller chunks for better embedding and retrieval
        texts = self.text_splitter.split_documents(documents)
        
        # Store the split texts in Chroma vector database
        return Chroma.from_documents(texts, self.embeddings)

    def ask(self, question: str) -> dict:
        """
        Ask a question about the codebase
        
        Args:
            question (str): The question to ask about the codebase
        
        Returns:
            dict: Contains the answer and source documents
        """
        return self.qa_chain.invoke({"question": question})  # Query the conversational retrieval chain

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Get OpenAI API key from environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Raise an error if the API key is missing
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY in your environment or .env file")
    
    # Ask the user for the repository path
    repo_path = input("Enter the path to your local git repository: ")
    
    # Create an instance of CodebaseChat
    chat = CodebaseChat(repo_path, api_key)
    
    print("\nCodebase Chat initialized! Ask questions about your code (type 'exit' to quit):")
    
    while True:
        question = input("\nYour question: ")  # Prompt user for a question
        if question.lower() == 'exit':  # Exit loop if user types 'exit'
            break
        
        try:
            result = chat.ask(question)  # Process user question through CodebaseChat
            print("\nAnswer:", result['answer'])  # Print the retrieved answer
            
            print("\nSources:")  # Print source files used to generate the answer
            for doc in result['source_documents']:
                print(f"- {doc.metadata['source']}")
        except Exception as e:
            print(f"Error: {e}")  # Print an error message if something goes wrong

# Run the main function when the script is executed
if __name__ == "__main__":
    main()
