from llama_index import StorageContext, load_index_from_storage
import os
import openai
from dotenv import load_dotenv
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, LLMPredictor, PromptHelper, LangchainEmbedding, ServiceContext, load_index_from_storage
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings


# Load environment variables (set OPENAI_API_KEY and OPENAI_API_BASE in .env)
load_dotenv()

# Configure OpenAI API
openai.api_type = "azure"
openai.api_version = os.getenv('OPENAI_API_VERSION')
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    client=openai.ChatCompletion(),
    model_name="gpt-4",
    temperature=0.7,
    max_tokens=1000,
    model_kwargs={"deployment_id": "gpt-4"},
)
llm_predictor = LLMPredictor(llm=llm)
embedding_llm = LangchainEmbedding(OpenAIEmbeddings(
    openai_api_base=os.getenv('OPENAI_API_BASE'),
    openai_api_type='azure',
    deployment='embedding',
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    chunk_size=1,
))

# Define prompt helper
max_input_size = 3000
num_output = 256
chunk_size_limit = 1000
max_chunk_overlap = 0.1
prompt_helper = PromptHelper(
    max_input_size, num_output, max_chunk_overlap, chunk_size_limit)

service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor, prompt_helper=prompt_helper, embed_model=embedding_llm)

# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir='./storage')
# load index
index = load_index_from_storage(
    storage_context, service_context=service_context)

# # check if index is loaded
if index is None:
    print("index is not loaded")
    documents = SimpleDirectoryReader('data').load_data()

    # Create index
    index = GPTVectorStoreIndex.from_documents(
        documents, service_context=service_context)
    # persist index
    index.storage_context.persist()

query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print("\nQuery response: ", response)
