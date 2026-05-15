import importlib
from typing import Optional

from outhad_contextkit.configs.embeddings.base import BaseEmbedderConfig
from outhad_contextkit.configs.llms.base import BaseLlmConfig
from outhad_contextkit.embeddings.mock import MockEmbeddings


def load_class(class_type):
    module_path, class_name = class_type.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class LlmFactory:
    provider_to_class = {
        "ollama": "outhad_contextkit.llms.ollama.OllamaLLM",
        "openai": "outhad_contextkit.llms.openai.OpenAILLM",
        "groq": "outhad_contextkit.llms.groq.GroqLLM",
        "together": "outhad_contextkit.llms.together.TogetherLLM",
        "aws_bedrock": "outhad_contextkit.llms.aws_bedrock.AWSBedrockLLM",
        "litellm": "outhad_contextkit.llms.litellm.LiteLLM",
        "azure_openai": "outhad_contextkit.llms.azure_openai.AzureOpenAILLM",
        "openai_structured": "outhad_contextkit.llms.openai_structured.OpenAIStructuredLLM",
        "anthropic": "outhad_contextkit.llms.anthropic.AnthropicLLM",
        "azure_openai_structured": "outhad_contextkit.llms.azure_openai_structured.AzureOpenAIStructuredLLM",
        "gemini": "outhad_contextkit.llms.gemini.GeminiLLM",
        "deepseek": "outhad_contextkit.llms.deepseek.DeepSeekLLM",
        "xai": "outhad_contextkit.llms.xai.XAILLM",
        "sarvam": "outhad_contextkit.llms.sarvam.SarvamLLM",
        "lmstudio": "outhad_contextkit.llms.lmstudio.LMStudioLLM",
        "vllm": "outhad_contextkit.llms.vllm.VllmLLM",
        "langchain": "outhad_contextkit.llms.langchain.LangchainLLM",
    }

    @classmethod
    def create(cls, provider_name, config):
        class_type = cls.provider_to_class.get(provider_name)
        if class_type:
            llm_instance = load_class(class_type)
            base_config = BaseLlmConfig(**config)
            return llm_instance(base_config)
        else:
            raise ValueError(f"Unsupported Llm provider: {provider_name}")


class EmbedderFactory:
    provider_to_class = {
        "openai": "outhad_contextkit.embeddings.openai.OpenAIEmbedding",
        "ollama": "outhad_contextkit.embeddings.ollama.OllamaEmbedding",
        "huggingface": "outhad_contextkit.embeddings.huggingface.HuggingFaceEmbedding",
        "azure_openai": "outhad_contextkit.embeddings.azure_openai.AzureOpenAIEmbedding",
        "gemini": "outhad_contextkit.embeddings.gemini.GoogleGenAIEmbedding",
        "vertexai": "outhad_contextkit.embeddings.vertexai.VertexAIEmbedding",
        "together": "outhad_contextkit.embeddings.together.TogetherEmbedding",
        "lmstudio": "outhad_contextkit.embeddings.lmstudio.LMStudioEmbedding",
        "langchain": "outhad_contextkit.embeddings.langchain.LangchainEmbedding",
        "aws_bedrock": "outhad_contextkit.embeddings.aws_bedrock.AWSBedrockEmbedding",
    }

    @classmethod
    def create(cls, provider_name, config, vector_config: Optional[dict]):
        if provider_name == "upstash_vector" and vector_config and vector_config.enable_embeddings:
            return MockEmbeddings()
        class_type = cls.provider_to_class.get(provider_name)
        if class_type:
            embedder_instance = load_class(class_type)
            base_config = BaseEmbedderConfig(**config)
            return embedder_instance(base_config)
        else:
            raise ValueError(f"Unsupported Embedder provider: {provider_name}")


class VectorStoreFactory:
    provider_to_class = {
        "qdrant": "outhad_contextkit.vector_stores.qdrant.Qdrant",
        "chroma": "outhad_contextkit.vector_stores.chroma.ChromaDB",
        "pgvector": "outhad_contextkit.vector_stores.pgvector.PGVector",
        "milvus": "outhad_contextkit.vector_stores.milvus.MilvusDB",
        "upstash_vector": "outhad_contextkit.vector_stores.upstash_vector.UpstashVector",
        "azure_ai_search": "outhad_contextkit.vector_stores.azure_ai_search.AzureAISearch",
        "pinecone": "outhad_contextkit.vector_stores.pinecone.PineconeDB",
        "mongodb": "outhad_contextkit.vector_stores.mongodb.MongoDB",
        "redis": "outhad_contextkit.vector_stores.redis.RedisDB",
        "elasticsearch": "outhad_contextkit.vector_stores.elasticsearch.ElasticsearchDB",
        "vertex_ai_vector_search": "outhad_contextkit.vector_stores.vertex_ai_vector_search.GoogleMatchingEngine",
        "opensearch": "outhad_contextkit.vector_stores.opensearch.OpenSearchDB",
        "supabase": "outhad_contextkit.vector_stores.supabase.Supabase",
        "weaviate": "outhad_contextkit.vector_stores.weaviate.Weaviate",
        "faiss": "outhad_contextkit.vector_stores.faiss.FAISS",
        "langchain": "outhad_contextkit.vector_stores.langchain.Langchain",
    }

    # Phase T3 — per-provider collection-name field override map. Most
    # providers expose a ``collection_name`` field (verified across all
    # configs/vector_stores/*.py); the few that diverge (Baidu uses
    # ``table_name``) are handled here so the factory caller does not
    # need provider-specific knowledge.
    _COLLECTION_FIELD = {
        "baidu": "table_name",
    }

    @classmethod
    def _collection_field(cls, provider_name: str) -> str:
        return cls._COLLECTION_FIELD.get(provider_name, "collection_name")

    @classmethod
    def create(cls, provider_name, config, *, collection_override: Optional[str] = None):
        """Instantiate a vector store.

        ``collection_override`` rebinds the provider's collection-name
        field (``collection_name`` for most; provider-specific entries
        in ``_COLLECTION_FIELD``) without mutating the input config —
        the override is applied on a shallow copy so per-tenant routing
        does not leak across calls.
        """
        class_type = cls.provider_to_class.get(provider_name)
        if class_type:
            if not isinstance(config, dict):
                config = config.model_dump()
            else:
                # Shallow copy so the override below never mutates the caller's dict.
                config = dict(config)
            if collection_override:
                field = cls._collection_field(provider_name)
                config[field] = collection_override
            vector_store_instance = load_class(class_type)
            return vector_store_instance(**config)
        else:
            raise ValueError(f"Unsupported VectorStore provider: {provider_name}")

    @classmethod
    def reset(cls, instance):
        instance.reset()
        return instance
