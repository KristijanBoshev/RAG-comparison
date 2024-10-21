from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import  RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from app.schema import Entities
from app.graph.KG import kgraph
from app.settings import settings

class Generation:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = self.prompt()
        self.chain = self.chain()
        
        
    def prompt(self):
        prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are extracting crucial entities from the text.",
        ),
        (
            "human",
            "Use the given format to extract information from the following "
            "input: {question}",
        ),
    ]
    )
        return prompt
    
    def chain(self):
        chain = self.llm.with_structured_output(Entities)
        return chain


    def graph_retriever(self, question: str) -> str:
        """
        Collects the neighborhood of entities mentioned
        in the question
        """
        result = ""
        entities = self.chain.invoke(question)
        for entity in entities.names:
            response = kgraph.graph.query(
                """
                CALL db.index.fulltext.queryNodes('fulltext_entity_id', $query, {limit:2})
                YIELD node, score
                CALL {
                    WITH node
                    MATCH (node)-[r]->(neighbor)
                    WHERE NOT type(r) = 'MENTIONS'
                    RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                    UNION ALL
                    WITH node
                    MATCH (node)<-[r]-(neighbor)
                    WHERE NOT type(r) = 'MENTIONS'
                    RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
                }
                RETURN output LIMIT 50
                """,
                {"query": entity},
            )
            result += "\n".join([el['output'] for el in response])
        return result
    
    def full_retriever(self, question: str):
        graph_data = self.graph_retriever(question)
        vector_data = [el.page_content for el in kgraph.vector_retriever.invoke(question)]
        final_data = f"""Graph data:
        {graph_data}
        vector data:
        {"#Document ". join(vector_data)}
        """
        return final_data
    
    def response_chain(self):
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        Use natural language and be concise.
        Answer:"""
        prompt = ChatPromptTemplate.from_template(template)

        chain = (
        {
            "context": self.full_retriever,
            "question": RunnablePassthrough(),
        }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain
    
    def run(self, query):
        chain = self.response_chain()
        response = chain.invoke(query)
        return response
                