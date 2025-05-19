import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genia
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

load_dotenv('.env')

genia.configure(api_key=os.getenv("GOOGLE_API_KEY"))

from src.agents.kanji_semantic_agent import KanjiSemanticAgent


HUMAN = "HUMAN"
AI = "AI"


class StreamlitUI:
    def __init__(self):
        load_dotenv('.env')
        self.__init_semantic_agent()


    def __init_semantic_agent(self):
        if "semantic_agent" not in st.session_state:

            gemini_llm = ChatGoogleGenerativeAI(
                model='models/gemini-2.0-flash',
            )
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

            st.session_state.semantic_agent = agent

    def display_sidebar(self):
        st.sidebar.image('./assets/siafi-logo-blanco.webp', use_container_width=True)
        st.sidebar.image('./assets/kanji-ji_logo2.png', use_container_width=True)


        st.sidebar.title("🍣 RAG-Sensei")

        st.sidebar.markdown(
            """
            Este agente forma parte del proyecto Kanji-Ji, una iniciativa diseñada para ayudar a los estudiantes a aprender kanji y vocabulario japonés.
            """
        )

        st.sidebar.markdown("Versión beta")

    def display_chat_history(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message['type']):
                st.markdown(message["content"])


    def handle_human_message(self, message: str):
        st.chat_message(HUMAN).markdown(message)

        st.session_state.messages.append(
            {
                "type": HUMAN,
                "content": message
            }
        )

    def handle_ai_message(self, prompt: str):
        agent = st.session_state.semantic_agent
        response = agent.generate_stream(prompt, 5)

        with st.chat_message(AI):
            chat_response = st.write_stream(response)

        st.session_state.messages.append(
            {
                "type": AI,
                "content": chat_response
            }
        )

    def run(self):
        st.title("🈹 RAG-Sensei")
        self.display_sidebar()
        st.markdown("Esta aplicación es un agente experto en Kanjis japoneses que responde preguntas y explica su significado, uso y etimología de manera clara y sencilla.")

        self.display_chat_history()

        prompt = st.chat_input("Pregunta algo a RAG-Sensei")
        if prompt:
            self.handle_human_message(prompt)
            self.handle_ai_message(prompt)


if __name__ == "__main__":
    ui = StreamlitUI()
    ui.run()