from app.naive.retrieval import retrieval
from langchain_core.prompts import ChatPromptTemplate


class Generation:
    def __init__(self, query, llm):
        self.query = query
        self.llm = llm

    def _create_prompt(self):
        template = """You are an AI assistant tasked with answering questions based on the provided context.
        
        Context: {context}
        
        Human: {question}
        
        AI: Let me answer that based on the information provided:
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        return prompt

    def _get_context(self):
        docs = retrieval._assign_retriever()
        context = "\n".join([doc.page_content for doc in docs])
        return context

    def generate_answer(self):
        context = self._get_context()
        prompt = self._create_prompt()
        
        chain = prompt | self.llm
        
        response = chain.invoke({"context": context, "question": self.query})
        return response