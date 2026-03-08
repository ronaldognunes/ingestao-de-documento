import os
import time
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH")
BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "5"))
MAX_RETRIES = int(os.getenv("EMBEDDING_MAX_RETRIES", "5"))
RETRY_BASE_SECONDS = int(os.getenv("EMBEDDING_RETRY_BASE_SECONDS", "40"))

def resolve_embedding_model() -> str:
    if os.getenv("MODEL").strip() == "GEMINI":
        model = (os.getenv("GOOGLE_EMBEDDING_MODEL") or "").strip()
        return model or "models/gemini-embedding-001"
    else:
        model = (os.getenv("OPENAI_EMBEDDING_MODEL") or "").strip()
        return model or "text-embedding-3-small"


EMBEDDING_MODEL = resolve_embedding_model()


def is_quota_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "429" in message or "resourceexhausted" in message or "quota" in message


def add_documents_with_retry(store: PGVector, documents: list):
    for start in range(0, len(documents), BATCH_SIZE):
        batch = documents[start:start + BATCH_SIZE]

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                store.add_documents(batch)
                print(f"✅ Lote {start // BATCH_SIZE + 1} inserido ({len(batch)} chunks)")
                break
            except Exception as exc:
                if not is_quota_error(exc) or attempt == MAX_RETRIES:
                    raise

                wait_seconds = RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                print(
                    f"⚠️  Cota/rate limit atingido (tentativa {attempt}/{MAX_RETRIES}). "
                    f"Aguardando {wait_seconds}s para retentar..."
                )
                time.sleep(wait_seconds)

def ingest_pdf():
    current_dir = Path(__file__).parent.parent
    pdf_path = current_dir / PDF_PATH
    docs = PyPDFLoader(str(pdf_path)).load()

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=False
    ).split_documents(docs)

    if not chunks:
        raise SystemExit(0)

    if os.getenv("MODEL").strip() == "GEMINI":
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    else:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    store = PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True,
    )

    add_documents_with_retry(store, chunks)

    print("✅ Embeddings armazenados com sucesso!")

if __name__ == "__main__":
    ingest_pdf()