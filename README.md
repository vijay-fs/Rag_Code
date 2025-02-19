# Codebase Chat

A tool that allows you to chat with and ask questions about your codebase using LangChain and OpenAI's LLMs.

## Setup

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the project root and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

Run the script:
```bash
python chat_with_code.py
```

When prompted, enter the path to your local git repository. Then you can start asking questions about the codebase.

The tool will:
1. Load and index all code files from the repository
2. Create embeddings using OpenAI's embedding model
3. Store the vectors in a Chroma vector database
4. Use GPT-3.5-turbo to answer your questions based on the relevant code context

Example questions you can ask:
- What are the main functions in this codebase?
- How is error handling implemented?
- Explain the architecture of this project
- What dependencies does this project use?

## Average cost per question

<img src="/docs//images//usage.png" alt="Average cost per question" />

Type 'exit' to quit the chat.

To know more about the project, visit the [documentation](/docs/README.md)