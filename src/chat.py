from search import search_prompt

def main():
    print("🤖 Chat RAG iniciado! (digite 'sair' para encerrar)\n")

    while True:
        
        pergunta = input("Faça sua pergunta: ")
        if pergunta.lower() in ["sair", "exit", "quit"]:
            break
        
        resposta = search_prompt(pergunta)
        
        if not resposta:
            print("Não foi possível iniciar o chat. Verifique os erros de inicialização. ")
            return

        print(f"IA: {resposta}\n")

if __name__ == "__main__":
    main()