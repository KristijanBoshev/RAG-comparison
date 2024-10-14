from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain import hub
from langchain_core.output_parsers import StrOutputParser

class Generate:
    def __init__(self, model, temperature):
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.rag_chain = None
        self.question_rewriter = None

    def create_grade_prompt(self):
        system = """You are a grader assessing relevance of a retrieved document to a user question. 
        If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
        Give a binary 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
            ]
        )
        retrieval_grader = grade_prompt | self.llm | StrOutputParser()
       
        return retrieval_grader
    
    def grade_documents(self, retriever, question):
        docs = retriever.invoke(question)
        doc_txt = docs[1].page_content  # Assuming grading the second document
        retrieval_grader = self.create_grade_prompt()
        result = retrieval_grader.invoke({"question": question, "document": doc_txt})
   
        return result

    def create_rag_chain(self):
        prompt = hub.pull("rlm/rag-prompt")
        self.rag_chain = prompt | self.llm | StrOutputParser()
   
        return self.rag_chain

    def run_rag_chain(self, docs, question) -> str:
        if self.rag_chain is None:
            self.create_rag_chain() 
        generation = self.rag_chain.invoke({"context": docs, "question": question})
      
        return generation

    def create_rewrite_prompt(self):
        system = """You are a question re-writer that converts an input question to a better version optimized for web search.
        Look at the input and try to reason about the underlying semantic intent / meaning."""
        
        re_write_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "Here is the initial question: \n\n {question} \n Formulate an improved question.",
                ),
            ]
        )
        self.question_rewriter = re_write_prompt | self.llm | StrOutputParser()
      
        return self.question_rewriter

    def rewrite_question(self, question):
        if self.question_rewriter is None:
            self.create_rewrite_prompt()
        rewritten_question = self.question_rewriter.invoke({"question": question})

        return rewritten_question
    
    def workflow(self, retriever, question):
        graded_result = self.grade_documents(retriever, question)
        docs = retriever.invoke(question)
        generation = self.run_rag_chain(docs, question)
        rewritten_question = self.rewrite_question(question)
     
        return {
            "graded_result": graded_result,
            "generation": generation,
            "rewritten_question": rewritten_question
        }