from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from app.settings import settings
from app.naive.ingestion import document_ingest
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sko.GA import GA
from sko.SA import SA
from sko.PSO import PSO
import optuna

collection_name = settings.CHROMA_COLLECTION_NAME
persistent_directory = settings.CHROMA_PERSIST_DIRECTORY

"""
This class retrieves information and tunes hyperparameters k and fetch_k.
"""
class Retrieval:
    def __init__(self, query):
        self.query = query
        self.embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=settings.OPENAI_API_KEY,
        )
        self.param_ranges = {'k': (1, 20), 'fetch_k': (10, 50)}

    def _create_vector_store(self):
        return Chroma.from_documents(
            collection_name=collection_name,
            persist_directory=persistent_directory,
            documents=document_ingest._loader(),
            embedding=document_ingest.embedding_model
        )

    def _assign_retriever(self, k, fetch_k):
        vector_store = self._create_vector_store()
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={'k': int(k), 'fetch_k': int(fetch_k)}
        )
        return retriever.invoke(self.query)

    def _evaluate(self, params):
        k, fetch_k = int(params[0]), int(params[1])
        docs = self._assign_retriever(k, fetch_k)
        docs = docs[:k]

        if not docs:
            return -1.0

        doc_texts = [doc.page_content for doc in docs]
        doc_embeddings = self.embedding_model.embed_documents(doc_texts)
        query_embedding = self.embedding_model.embed_query(self.query)
        cosine_sim = cosine_similarity([query_embedding], doc_embeddings)[0]
        return float(np.max(cosine_sim))

    def tune_with_optuna(self, n_trials=10):
        def objective(trial):
            k = trial.suggest_int('k', self.param_ranges['k'][0], self.param_ranges['k'][1])
            fetch_k = trial.suggest_int('fetch_k', self.param_ranges['fetch_k'][0], self.param_ranges['fetch_k'][1])
            return self._evaluate([k, fetch_k])

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
        print(f"Optuna Best k: {best_params['k']}, Best fetch_k: {best_params['fetch_k']}")
        return best_params['k'], best_params['fetch_k']

    def tune_with_ga(self, n_iter=10, size_pop=10):
        def objective_function(params):
            return self._evaluate(params)

        ga = GA(func=objective_function, n_dim=2, size_pop=size_pop, max_iter=n_iter,
                lb=[self.param_ranges['k'][0], self.param_ranges['fetch_k'][0]],
                ub=[self.param_ranges['k'][1], self.param_ranges['fetch_k'][1]],
                precision=1)
        best_params, best_score = ga.run()
        best_k, best_fetch_k = int(best_params[0]), int(best_params[1])
        print(f"GA Best k: {best_k}, Best fetch_k: {best_fetch_k}, Best Score: {best_score}")
        return best_k, best_fetch_k

    def tune_with_sa(self, n_iter=10):
        def objective_function(params):
            k, fetch_k = int(params[0]), int(params[1])
            print(f"SA evaluating: k={params[0]}, fetch_k={params[1]}")
            if fetch_k < k or fetch_k < 1:  # Penalize invalid fetch_k values
                return -1e9
            return -self._evaluate([k, fetch_k])  # Negate for maximization

        # Refine x0 to be closer to the middle of the ranges
        x0 = [
            (self.param_ranges["k"][0] + self.param_ranges["k"][1]) // 2,
            (self.param_ranges["fetch_k"][0] + self.param_ranges["fetch_k"][1]) // 2,
        ]

        sa = SA(
            func=objective_function,
            x0=x0,
            T_max=1,
            T_min=1e-9,
            L=10,
            max_stay_counter=15,
        )
        best_params, best_score = sa.run()
        best_k, best_fetch_k = int(best_params[0]), int(best_params[1])
        print(
            f"SA Best k: {best_k}, Best fetch_k: {best_fetch_k}, Best Score: {-best_score}"
        )
        return best_k, best_fetch_k

    def tune_with_pso(self, n_iter=10, size_pop=10):
        def objective_function(params):
            return self._evaluate(params)

        pso = PSO(func=objective_function, n_dim=2, pop=size_pop, max_iter=n_iter,
                  lb=[self.param_ranges['k'][0], self.param_ranges['fetch_k'][0]],
                  ub=[self.param_ranges['k'][1], self.param_ranges['fetch_k'][1]])
        best_params, best_score = pso.run()
        best_k, best_fetch_k = int(best_params[0]), int(best_params[1])
        print(f"PSO Best k: {best_k}, Best fetch_k: {best_fetch_k}, Best Score: {best_score}")
        return best_k, best_fetch_k

    def tune_and_retrieve(self, tuning_technique):
        if tuning_technique == 'optuna':
            best_k, best_fetch_k = self.tune_with_optuna()
        elif tuning_technique == 'ga':
            best_k, best_fetch_k = self.tune_with_ga()
        elif tuning_technique == 'sa':
            best_k, best_fetch_k = self.tune_with_sa()
        elif tuning_technique == 'pso':
            best_k, best_fetch_k = self.tune_with_pso()
        else:
            raise ValueError(f"Unknown tuning technique: {tuning_technique}")

        return self._assign_retriever(best_k, best_fetch_k)