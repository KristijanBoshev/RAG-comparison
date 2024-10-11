from app.naive.generation import generate
from app.naive.ingestion import document_ingest

def main():
   
    generate_answer = generate._generate_answer()

    print(f'Answer: {generate_answer.content}')

if __name__ == '__main__':
    main()    