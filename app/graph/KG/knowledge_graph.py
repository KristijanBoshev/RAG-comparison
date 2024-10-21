from neo4j import Driver
from neo4j import GraphDatabase
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_openai.embeddings import OpenAIEmbeddings

class KGraph:
    def __init__(self, uri, username, password):
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = None
        self.graph = None
        self.vector_index = None
        self.vector_retriever = None
        self.connect()
        self.run()
        self.create_vector_index()

    def connect(self):
        self.graph = Neo4jGraph()
        with GraphDatabase.driver(
            uri=self.uri,
            auth=(self.username, self.password)
        ) as self.driver:
            self.driver.verify_connectivity()
        return self.graph, self.driver

    def close(self):
        if self.driver:
            self.driver.close()

    def add_graph_documents(self, graph_documents):
        self.graph.add_graph_documents(
        graph_documents,
        baseEntityLabel=True,
        include_source=True
        )
        
    def create_vector_index(self, model="text-embedding-3-large"):
        embeddings = OpenAIEmbeddings(model=model)
        self.vector_index = Neo4jVector.from_existing_graph(
            embeddings,
            search_type="hybrid",
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding"
        )
        self.vector_retriever = self.vector_index.as_retriever()
        return self.vector_index, self.vector_retriever
    
    def create_fulltext_index(self):
        check_query = '''
        SHOW INDEXES
        YIELD name, type
        WHERE name = 'fulltext_entity_id' AND type = 'FULLTEXT'
        '''
        
        query = '''
        CREATE FULLTEXT INDEX `fulltext_entity_id` 
        FOR (n:__Entity__) 
        ON EACH [n.id];
        '''
        with self.driver.session() as session:
            try:
                result = session.run(check_query)
                if result.single():
                    print("Fulltext index already created.")
                else: 
                    session.run(query)
                    print("Fulltext index created successfully.")    

            except Exception as e:
                print(f"Error creating fulltext index: {e}")
                
    def run(self):
        try:
            self.create_fulltext_index()
        except: 
            pass
        
        self.close()
        
            
                        