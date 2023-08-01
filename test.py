from llama_index import StorageContext, load_index_from_storage
import os
import openai
from dotenv import load_dotenv
from llama_index import VectorStoreIndex, PromptHelper, LangchainEmbedding, ServiceContext, load_index_from_storage
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from llama_index.callbacks import TokenCountingHandler, CallbackManager

from llama_index.node_parser import SimpleNodeParser
from llama_index.langchain_helpers.text_splitter import SentenceSplitter

from llama_hub.file.cjk_pdf.base import CJKPDFReader
from pathlib import Path
from llama_index import download_loader

from llama_index import get_response_synthesizer
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.postprocessor import SimilarityPostprocessor

from callbackHandler import CallbackHandler

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


# Create an instance of the TokenCountingHandler
token_counter = TokenCountingHandler()

# Add the TokenCountingHandler to the CallbackManager
callback_manager = CallbackManager([token_counter, CallbackHandler([], [])])

text_splitter = SentenceSplitter(
    separator="。",
    chunk_size=1024,
    chunk_overlap=20,
    backup_separators=["\n"],
    paragraph_separator="\n\n\n"
)
node_parser = SimpleNodeParser(text_splitter=text_splitter)

service_context = ServiceContext.from_defaults(
    llm=llm, prompt_helper=prompt_helper, embed_model=embedding_llm, callback_manager=callback_manager, node_parser=node_parser)

# check if storage is present
if not os.path.exists('./storage'):
    print("index is not present or loaded")
    # documents = SimpleDirectoryReader('data').load_data()
    CJKPDFReader = download_loader("CJKPDFReader")
    loader = CJKPDFReader()
    documents = loader.load_data(
        file=Path('./data/Colorectal-July22C-Revamp-Web.pdf'))  # way to load all data in a directory

    # Create index
    index = VectorStoreIndex.from_documents(
        documents, service_context=service_context, show_progress=True)
    # persist index
    index.storage_context.persist()
else:
    # rebuild storage context
    storage_context = StorageContext.from_defaults(
        persist_dir='./storage')
    # load index
    index = load_index_from_storage(
        storage_context, service_context=service_context)

# # configure retriever
# retriever = VectorIndexRetriever(
#     index=index,
#     similarity_top_k=2,
# )

# # configure response synthesizer
# response_synthesizer = get_response_synthesizer()
# synthesizer = get_response_synthesizer(
#     service_context=,
#     text_qa_template=,
#     refine_template=,
#     simple_template=,
#     response_mode=,
#     streaming=,
#     callback_manager=,
#     use_async=,)


# # assemble query engine
# query_engine = RetrieverQueryEngine(
#     retriever=retriever,
#     response_synthesizer=response_synthesizer,
#     node_postprocessors=[
#         SimilarityPostprocessor(similarity_cutoff=0.7)
#     ]
# )

query_engine = index.as_query_engine()  # response_mode="simple_summarize"
response = query_engine.query("用中文回答，大腸癌的成因")
# response = query_engine.query("what is the cause of colorectal cancer?")
# response = query_engine.query("what is colorectal cancer?")
print("\nQuery response: ", response)

# Get the token counts
# embedding_tokens = token_counter.total_embedding_token_count
# llm_prompt_tokens = token_counter.prompt_llm_token_count
# llm_completion_tokens = token_counter.completion_llm_token_count
# total_llm_tokens = token_counter.total_llm_token_count
# print("embedding_tokens: ", embedding_tokens)
# print("llm_prompt_tokens: ", llm_prompt_tokens)
# print("llm_completion_tokens: ", llm_completion_tokens)
# print("total_llm_tokens: ", total_llm_tokens)
