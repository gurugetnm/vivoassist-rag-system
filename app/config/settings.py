import os
from dataclasses import dataclass
from dotenv import load_dotenv

from llama_index.core import Settings
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding


@dataclass(frozen=True)
class AppConfig:
    # Paths
    data_dir: str = "./data/manuals"
    chroma_dir: str = "./chroma_db"
    chroma_collection: str = "pdf_chunks"

    # Retrieval
    top_k: int = 8
    debug: bool = True

    # Chunking (hierarchical)
    big_chunk_size: int = 1500
    big_chunk_overlap: int = 150
    mid_chunk_size: int = 800
    mid_chunk_overlap: int = 100
    small_chunk_size: int = 300
    small_chunk_overlap: int = 50


def load_config() -> AppConfig:
    """
    Loads .env variables and config constants.
    """
    load_dotenv()

    # Validate required env vars early (clear errors)
    required = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_CHAT_DEPLOYMENT",
        "AZURE_OPENAI_EMBED_DEPLOYMENT",
    ]
    missing = [k for k in required if k not in os.environ or not os.environ[k].strip()]
    if missing:
        raise RuntimeError(f"Missing required environment variables in .env: {missing}")

    return AppConfig()


def configure_llamaindex():
    """
    Configure LlamaIndex global Settings.llm and Settings.embed_model using Azure OpenAI.
    Call once at startup.
    """
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    api_key = os.environ["AZURE_OPENAI_API_KEY"]
    api_version = os.environ["AZURE_OPENAI_API_VERSION"]
    chat_deployment = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]
    embed_deployment = os.environ["AZURE_OPENAI_EMBED_DEPLOYMENT"]

    Settings.llm = AzureOpenAI(
        engine=chat_deployment,  # Azure deployment name
        model="gpt-4o",
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        temperature=0.2,
    )

    Settings.embed_model = AzureOpenAIEmbedding(
        engine=embed_deployment,  # Azure deployment name
        model="text-embedding-3-large",
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )
