from app.naive.generation import generate

def main():
    generate_answer = generate.generate_answer()

    print(f'Answer: {generate_answer}')

if __name__ == '__main__':
    main()    