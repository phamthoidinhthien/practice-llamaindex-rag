# Llama Index Blog Crawler and Query Engine

This project using provides a tool to crawl blog posts and utilize Llama-index Q&A LLama-index blog post via command-line interface. It supports re-crawling the blog, re-indexing, and running evaluations with optional retry support.

## Features

- Crawl blog posts and store them in a local database.
- Initialize and load an index from local storage.
- Query the index using a command-line interface.
- Optional evaluation mode with retry support for generating answers.

## Requirements

- Python 3.9
- Required Python packages (install via `requirements.txt`)

## Installation
Due to the upload size, please self install the environment

Install python3.9 & the required packages:
```sh
brew install python@3.9
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### API-KEY
This project use openAI, you must store OPENAI_API_KEY to .env file

```
OPENAI_API_KEY=
```

### Command-Line Arguments

- `--re-crawl`: Re-crawl the blog before running queries.
- `--eval`: Run evaluation on the test set.

### Running the Script

To run the script, use the following command:
```sh
python main.py [--re-crawl] [--eval]
```

### Example
1. Interact with the RAG using existing DB 
```sh
python main.py
```

1. Re-crawl the blog and re-index
```sh
python main.py --re-crawl
```

3. Run evaluation using RAGAS
```sh
python main.py --eval
```

