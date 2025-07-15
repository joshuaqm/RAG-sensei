# RAG-Sensei

Aplicación RAG (Retrieval-Augmented Generation) para el aprendizaje de kanjis japoneses, desarrollada para los proyectos **Kanji-Ji** y **SIAFI**.

La aplicación utiliza modelos de lenguaje, embeddings y bases de conocimiento para responder preguntas sobre kanjis con un enfoque semántico inteligente.

---

## 🚀 Características

- Generación aumentada con recuperación (RAG)
- Embeddings con modelo `mxbai-embed-large` de **Ollama**
- Interfaz web interactiva con **Streamlit**
- Almacenamiento vectorial con **ChromaDB**
- Configuración flexible con archivo `.env`

---

## 🛠️ Requisitos previos

Antes de instalar y ejecutar la aplicación, asegúrate de tener lo siguiente:

- Python 3.10+
- [Ollama](https://ollama.com/) instalado y en ejecución
- Modelo `mxbai-embed-large` descargado:
  ```bash
  ollama run mxbai-embed-large
