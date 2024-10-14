from app.naive.generation import generate
from app.agentic.workflow import crag
from app.settings import settings

def main():
   
    generate_answer = generate._generate_answer()

    print(f'Naive Answer: {generate_answer.content}')
    
    initial_state = {
        "question": settings.QUERY
    }
    
    crag.run(initial_state=initial_state)
    
    
    

if __name__ == '__main__':
    main()    