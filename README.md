# WebRAG Assistant 📊

WebRAG Assistant is a powerful tool that enables conversational interaction with website content using cutting-edge AI and Retrieval-Augmented Generation (RAG) technology. Simply provide a URL, and our tool will crawl, index, and allow you to ask questions about any website's content.



https://github.com/user-attachments/assets/8f8d17bb-9bde-4f41-803d-df4084adfb1a

![Image](https://github.com/user-attachments/assets/d4ea6d89-1327-4ec4-a62f-7b1e0a3d7b21)

![Image](https://github.com/user-attachments/assets/821a72c6-7d7c-472e-8967-06dd872f1b9d)


## Features

- 🕸️ **Crawl any website** - Extract clean, structured content from websites
- 🧠 **Vector-based retrieval** - Find the most relevant content to answer questions
- 💬 **Natural conversation** - Ask questions in natural language
- 📊 **Source citation** - See exactly where information comes from
- 🚀 **Fast setup** - Get started with just your API keys and a URL

## How It Works

WebRAG Assistant uses a **Retrieval-Augmented Generation (RAG)** pipeline to provide accurate, source-cited answers about website content:

1. **Crawling**: When you input a URL, the system crawls the website and extracts clean content
2. **Chunking**: Large pages are split into smaller, manageable chunks
3. **Embedding**: Text chunks are converted to vector embeddings using Sentence Transformers
4. **Storage**: Vectors are stored in a ChromaDB database for efficient retrieval
5. **Retrieval**: Your questions are matched with the most relevant content
6. **Generation**: Groq's Llama 3 LLM generates answers based solely on the retrieved content

## Tech Stack

### FireCrawl
FireCrawl handles the complex task of web crawling and content extraction. Unlike basic scrapers, FireCrawl:
- Handles JavaScript-rendered websites
- Extracts clean, structured content (removing navigation, ads, etc.)
- Follows links to discover additional pages
- Converts HTML to markdown format optimized for LLMs

### Vector Embeddings
Text embeddings are numerical representations of text that capture semantic meaning:
- We use Sentence Transformers to convert text into 384-dimensional vectors
- Similar text chunks have vectors that are close together in vector space
- This enables efficient semantic search beyond simple keyword matching

### ChromaDB
ChromaDB is a lightweight vector database that:
- Stores vector embeddings and their associated text
- Enables efficient similarity search to find the most relevant content
- Maintains persistence between sessions
- Scales to handle large websites with thousands of text chunks

### LangChain
LangChain provides the framework for connecting all components:
- Orchestrates the entire RAG pipeline
- Handles text splitting and processing
- Connects the vector store with the LLM
- Provides the infrastructure for retrieval-based generation

### Groq LLM (Llama 3)
Groq's implementation of Llama 3 powers the answer generation:
- Generates natural, coherent responses from retrieved content
- Maintains context awareness to create relevant answers
- Offers fast inference times compared to other LLM providers
- Balances accuracy and creativity in responses

## Installation

### Prerequisites
- Python 3.8+
- API keys for [Groq](https://console.groq.com) and [FireCrawl](https://firecrawl.dev)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Zakeertech3/WebRag_Assistant.git
cd webRag_assistant
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

5. Enter your API keys in the web interface when prompted

## Usage

1. Enter your Groq and FireCrawl API keys in the setup section
2. Enter a website URL and click "Initialize"
3. Wait for the website to be crawled and indexed
4. Ask questions about the website content in natural language
5. View source citations to see where information comes from

## Example Questions

After indexing a website, try asking:
- "What is this website about?"
- "What products or services do they offer?"
- "When was the company founded?"
- "What are their main features?"
- "How can I contact them?"

## Project Structure

```
webrag_assistant/
├── app.py                # Streamlit web interface
├── rag_pipeline.py       # Main RAG pipeline orchestration
├── crawler.py            # FireCrawl implementation
├── vector_store.py       # ChromaDB integration
├── llm_integration.py    # Groq LLM integration
├── config.py             # Configuration settings
├── requirements.txt      # Dependencies
└── README.md             # Documentation
```

## Limitations

- Only crawls public web pages (no authentication support)
- Limited to 20 pages per website by default (configurable)
- Response quality depends on website content quality
- Cannot answer questions beyond the content of the website

## Future Improvements

- Support for website authentication
- PDF and document file parsing
- Multi-website querying
- Custom crawling rules
- Advanced visualization of source relationships

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [FireCrawl](https://firecrawl.dev) for the web crawling capabilities
- [Groq](https://groq.com) for the LLM API
- [LangChain](https://python.langchain.com) for the RAG framework
- [ChromaDB](https://www.trychroma.com) for the vector database
- [Sentence Transformers](https://www.sbert.net) for the embedding model
- [Streamlit](https://streamlit.io) for the web interface

---

**Note**: This project is for educational and research purposes only. Please respect website terms of service and robots.txt files when crawling websites.
