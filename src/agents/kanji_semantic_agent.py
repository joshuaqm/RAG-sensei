from langchain.vectorstores import VectorStore
from langchain_core.prompts import PromptTemplate

_PROMPT = PromptTemplate.from_template("""
    Eres un experto en Kanjis japoneses, especializado en explicar su significado, uso y etimología de manera clara y sencilla. Utiliza los siguientes elementos del contexto recuperado para responder a la pregunta. No menciones al usuario que estás usando información recuperada. responde de manera clara, natural y amigable, como si estuvieras explicando a alguien que quiere aprender. 

    Pregunta: {question}
    Contexto: {context} 
                                       
    Respuesta:
""") 

class KanjiSemanticAgent:
    def __init__(self, vectorstore: VectorStore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
        
    def generate(self, question: str, k: int):
        similarity_results = self.vectorstore.similarity_search(question, k)
        prompt_with_context = _PROMPT.invoke(
            {
                'question': question,
                'context': similarity_results
            }
        )
        return self.llm.invoke(prompt_with_context)
    
    def generate_stream(self, question: str, k: int):
        similarity_results = self.vectorstore.similarity_search(question, k)
        prompt_with_context = _PROMPT.invoke(
            {
                'question': question,
                'context': similarity_results
            }
        )

        return self.llm.stream(prompt_with_context)