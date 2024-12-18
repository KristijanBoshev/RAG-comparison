from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os
from typing import List

load_dotenv()


class Settings(BaseSettings):
    # Key configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")

    # Query configuration
    QUERY: str = "Write comprehensive answer on what is multi-head attention and main benefits using it"
   
    # File configuration
    FILE_PATH: str = "/Users/kristijanboshev/Library/GitHub/RAG-comparison/data/Attention.pdf"

    # Chroma configuration
    CHROMA_COLLECTION_NAME: str = 'collection'
    CHROMA_PERSIST_DIRECTORY: str = './chroma_db'

    # LLM configuration
    MODEL: str = 'gpt-4o-mini'
    
    # Search engine
    SEARCH_ENGINE: str = 'TAVILY'
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY")
    URLS: List[str] = ["https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853",
            "https://medium.com/@sachinsoni600517/multi-head-attention-in-transformers-1dd087e05d41",
            ]
    USER_AGENT:str = os.getenv('USER_AGENT')
    # Neo4j credentials
    NEO4J_URI: str = os.getenv("NEO4J_URI")
    NEO4J_USERNAME:str = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD")
    AURA_INSTANCEID: str = os.getenv("AURA_INSTANCEID")
    AURA_INSTANCENAME:str = os.getenv("AURA_INSTANCENAME")
    
    GROUND_TRUTH: str = """Instead of performing a single attention function with dmodel-dimensional keys, values and queries, \
we found it beneficial to linearly project the queries, keys and values h times with different, learned \
linear projections to dk, dk and dv dimensions, respectively. On each of these projected versions of \
queries, keys and values we then perform the attention function in parallel, yielding dv-dimensional \
output values. These are concatenated and once again projected, resulting in the final values. \
The Transformer uses multi-head attention in three different ways:

• In "encoder-decoder attention" layers, the queries come from the previous decoder layer, \
and the memory keys and values come from the output of the encoder. This allows every \
position in the decoder to attend over all positions in the input sequence. This mimics the \
typical encoder-decoder attention mechanisms in sequence-to-sequence models such as \
[38, 2, 9].

• The encoder contains self-attention layers. In a self-attention layer all of the keys, values \
and queries come from the same place, in this case, the output of the previous layer in the \
encoder. Each position in the encoder can attend to all positions in the previous layer of the \
encoder.

• Similarly, self-attention layers in the decoder allow each position in the decoder to attend to \
all positions in the decoder up to and including that position. We need to prevent leftward \
information flow in the decoder to preserve the auto-regressive property. We implement this \
inside of scaled dot-product attention by masking out (setting to −∞) all values in the input \
of the softmax which correspond to illegal connections."""


    class Config:
        env_file_encoding = "utf-8"  # Ensure the env file is read with UTF-8 encoding
        
        


settings = Settings()


