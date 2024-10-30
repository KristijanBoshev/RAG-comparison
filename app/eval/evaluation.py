from datetime import datetime
import os
import pandas as pd
from app.settings import settings
from ragas import EvaluationDataset, evaluate
from ragas.metrics import FactualCorrectness,SemanticSimilarity
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


class Evaluate:
    def __init__(self):
        self.evaluator_llm = LangchainLLMWrapper(
            ChatOpenAI(model="gpt-4o-mini")
        )
        self.evaluator_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings()
        )
        
    def get_content_string(self, response):
        """
        Extract string content from various response types.
        """
        if hasattr(response, 'content'): 
            return response.content
        elif isinstance(response, str):
            return response
        else:
            return str(response)
            
    
    def create_dataset(self, naive_answer, kg_answer):
        """
        Create dataset from answers.
        """
        naive_response = self.get_content_string(naive_answer)
        kg_response = self.get_content_string(kg_answer)
        
        dataset = [
            {
                "user_input": settings.QUERY,
                "response": naive_response,
                "reference": settings.GROUND_TRUTH
            },
            {
                "user_input": settings.QUERY,
                "response": kg_response,
                "reference": settings.GROUND_TRUTH
            }
        ]
        
 
        return dataset

    def evaluate_answers(self, naive_answer, kg_answer):
        """
        Evaluate answers using RAGAS metrics.
        """
        dataset = self.create_dataset(naive_answer, kg_answer)
        
        eval_dataset = EvaluationDataset.from_list(dataset)
        
        metrics = [
            FactualCorrectness(llm=self.evaluator_llm),
            SemanticSimilarity(embeddings=self.evaluator_embeddings)
        ]
        
        results = evaluate(
            eval_dataset,
            metrics=metrics
        )

        detailed_results = results.to_pandas()
        
        detailed_results.index = ['Naive', 'Knowledge Graph']
        detailed_results.index.name = 'Approach'
        
        return detailed_results
