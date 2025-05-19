from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
import os
import google.generativeai as genia

from src.agents.kanji_semantic_agent import KanjiSemanticAgent

load_dotenv('.env')

genia.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_llm = ChatGoogleGenerativeAI(
    model='models/gemini-2.0-flash',
)

question = 'Qué es un kanji?'
embeddings_generator = OllamaEmbeddings(
    model='mxbai-embed-large'
)

vector_store = Chroma(
    collection_name='pokemon',
    embedding_function=embeddings_generator,
    persist_directory='./database/chromadb'
)

agent = KanjiSemanticAgent(
    vector_store,
    gemini_llm
)

response = agent.generate_stream(question, k=5)

for token in response:
    print(token.content, end='', flush=True)