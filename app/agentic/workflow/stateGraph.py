from langchain.schema import Document
from langgraph.graph import END, StateGraph, START
from app.agentic.state import GraphState
from app.agentic.search_tool import web_search_tool
from app.agentic.indexing import index
from app.agentic.llm import generate
from pprint import pprint


class CRAG:
    def __init__(self):
        self.workflow = StateGraph(GraphState)
        self.retriever = index.workflow()
        self.setup_graph()

    def retrieve(self, state):
        print("---RETRIEVE---")
        question = state["question"]
        documents = self.retriever.invoke(question)
        new_state = {"documents": documents, "question": question}
        return new_state

    def generate(self, state):
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        generation = generate.workflow(self.retriever, question)
        new_state = {"documents": documents, "question": question, "generation": generation}
        return new_state

    def grade_documents(self, state):
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        filtered_docs = []
        web_search = "No"
        for d in documents:
            score = generate.grade_documents(self.retriever, question)
            # Change here: Check if the score is 'yes' (case-insensitive)
            grade = score.lower() == 'yes'
            if grade:
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                web_search = "Yes"

        new_state = {"documents": filtered_docs, "question": question, "web_search": web_search}
        return new_state

    def transform_query(self, state):
        print("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]
        better_question = generate.rewrite_question(question)
        new_state = {"documents": documents, "question": better_question}
        return new_state

    def web_search(self, state):
        print("---WEB SEARCH---")
        question = state["question"]
        documents = state["documents"]

        docs = web_search_tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)
        documents.append(web_results)

        new_state = {"documents": documents, "question": question}
        return new_state

    def decide_to_generate(self, state):
        print("---ASSESS GRADED DOCUMENTS---")
        web_search = state.get("web_search", "No")
        if web_search == "Yes":
            return "transform_query"
        return "generate"

    def setup_graph(self):
        self.workflow.add_node("retrieve", self.retrieve)
        self.workflow.add_node("grade_documents", self.grade_documents)
        self.workflow.add_node("generate", self.generate)
        self.workflow.add_node("transform_query", self.transform_query)
        self.workflow.add_node("web_search_node", self.web_search)

        self.workflow.add_edge(START, "retrieve")
        self.workflow.add_edge("retrieve", "grade_documents")
        self.workflow.add_conditional_edges(
        "grade_documents",
        self.decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
        self.workflow.add_edge("transform_query", "web_search_node")
        self.workflow.add_edge("web_search_node", "generate")
        self.workflow.add_edge("generate", END)

    def run(self, initial_state):
        app = self.workflow.compile()
        # Run the graph
        final_state = app.invoke(initial_state)
    
        for output in final_state.items():
            pprint(f"Node '{output[0]}': {output[1]}")
    
    # Check if 'generation' is in the output, otherwise provide a default message
        if "generation" in final_state:
            pprint(final_state["generation"])
        else:
            pprint("No generation produced. Check the graph execution.")