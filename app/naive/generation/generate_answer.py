from app.naive.retrieval import retrieval
from langchain_core.prompts import ChatPromptTemplate

"""
Class responsible for final generation
"""

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

    def _get_context(self, tuning_technique):
        docs = retrieval.tune_and_retrieve(tuning_technique=tuning_technique)
        context = "\n".join([doc.page_content for doc in docs])
        return context

    def _generate_answer(self, tuning_technique='optuna'):
        context = self._get_context(tuning_technique=tuning_technique)
        prompt = self._create_prompt()
        chain = prompt | self.llm
        response = chain.invoke({"context": context, "question": self.query})
        return response