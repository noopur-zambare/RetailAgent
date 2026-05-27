# RetailAgent
AI-powered Canadian Retail Intelligence System using Statistics Canada datasets, SQLite, and LLM-based reasoning. RetailAgent is an intelligent business analytics assistant that helps users explore and understand Canadian retail trade data using natural language.

It combines:

- 📊 Structured SQL database (SQLite)
- 🧠 LLM reasoning (Gemini)
- 🔍 Retrieval layer (RAG-ready design)
- 📈 Retail analytics (monthly + annual Statistics Canada Data)
- 💬 Chat-based interface (Streamlit)


## Features
- Ask questions in natural language about Canadian retail trends
- Analyze monthly retail sales by industry
- Compare annual retail performance metrics
- LLM-based explanations and insights

## System Architecture

RetailAgent combines multiple layers:

- **SQLite Database** - Stores structured retail sales data  
- **ChromaDB Vector Store** - Enables semantic retrieval over rows  
- **Gemini LLM** - Generates analytical insights from retrieved context  
- **Streamlit UI** - Provides chat-based user interaction  

## Datasets
1. Statistics Canada. Table 20-10-0056-02  Monthly retail trade sales by industry (x 1,000)
2. Statistics Canada. Table 20-10-0083-01  Annual retail trade survey, summary statistics
3. Statistics Canada. Table 20-10-0084-01  Annual retail trade survey, sales (x 1,000)

## Tech Stack
- Frontend: Streamlit
- AI Model: Google Gemini 2.5 Flash
- Vector Database: ChromaDB


## ⚠️ Disclaimer
This application provides information for educational purposes only. 