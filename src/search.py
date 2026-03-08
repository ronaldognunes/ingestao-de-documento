import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

def resolve_embedding_model() -> str:
    if os.getenv("MODEL").strip() == "GEMINI":
        model = (os.getenv("GOOGLE_EMBEDDING_MODEL") or "").strip()
        return model or "models/gemini-embedding-001"
    else:
        model = (os.getenv("OPENAI_EMBEDDING_MODEL") or "").strip()
        return model or "text-embedding-3-small"


EMBEDDING_MODEL = resolve_embedding_model()


PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

def search_prompt(question=None):
    
    if os.getenv("MODEL").strip() == "GEMINI":
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0.2,
        )
    else:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
        )  

    store = PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True,
    )

    if not question:
        return None

    docs = store.similarity_search(question, k=10)

    prompt = PromptTemplate(
        input_variables=["contexto", "pergunta"],
        template=PROMPT_TEMPLATE,
    )

    chain = prompt | llm

    contexto = "\n\n".join([d.page_content for d in docs])

    resposta = chain.invoke({"contexto": contexto, "pergunta": question})

    return resposta.content if hasattr(resposta, "content") else str(resposta)