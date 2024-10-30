import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from app.naive.generation import generation
from app.agentic.workflow import crag
from app.settings import settings
from app.graph.generate import generate
from app.eval import evaluate



def main():

    naive_generated_answer = generation._generate_answer()
    print(f'Naive Answer: {naive_generated_answer.content}')
    
    initial_state = {
        "question": settings.QUERY
    }
    
    crag_generated_answer =crag.run(initial_state=initial_state)
    print(f'CRAG Answer: {crag_generated_answer}')
    
    kg_generated_answer = generate.run(query=settings.QUERY)
    print("Knowledge Graph:  ", kg_generated_answer)
    
    results = evaluate.evaluate_answers(naive_generated_answer, kg_generated_answer)
    print(results)

    kg = generate.run(query=settings.QUERY)
    print("Knowledge Graph:  ", kg)

  

if __name__ == '__main__':
    main()    