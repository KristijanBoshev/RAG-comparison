from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from app.settings import settings
from app.naive.ingestion import document_ingest
import optuna
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


collection_name = settings.CHROMA_COLLECTION_NAME
persistent_directory = settings.CHROMA_PERSIST_DIRECTORY

"""
This class is responsible for the retrieving process, including using optuna
for bayes hyperparameter tuning
"""

class Retrieval:
    def __init__(self, query):
        self.query = query
        self.embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=settings.OPENAI_API_KEY,
        )

    def _create_vector_store(self):
        vector_store = Chroma.from_documents(
            collection_name=collection_name,
            persist_directory=persistent_directory,
            documents= document_ingest._loader(),
            embedding=document_ingest.embedding_model
        )
        return vector_store
    
    def _assign_retriever(self, k, fetch_k):
      
        vector_store = self._create_vector_store()
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={'k': k, 'fetch_k': fetch_k}
        )
        docs = retriever.invoke(self.query)
        return docs
    
    
    def _objective(self, trial):
     
        k = trial.suggest_int('k', 1, 20)  
        fetch_k = trial.suggest_int('fetch_k', 10, 50)  
        docs = self._assign_retriever(k, fetch_k)
        
        docs = docs[:k]
    
        if isinstance(docs[0], str):
            doc_texts = docs
        else:
            doc_texts = [doc.page_content for doc in docs]
    
        doc_embeddings = self.embedding_model.embed_documents(doc_texts)
    
        query_embedding = self.embedding_model.embed_query(self.query)
        cosine_sim = cosine_similarity([query_embedding], doc_embeddings)[0]

        return float(np.max(cosine_sim))

    def tune_retriever(self, n_trials=5):
       
        study = optuna.create_study(direction='maximize')
        study.optimize(self._objective, n_trials=n_trials)
        
        print("Best Hyperparameters: ", study.best_params)
        print("Best Value: ", study.best_value)
        return self._assign_retriever(study.best_params['k'], study.best_params['fetch_k'])
                

