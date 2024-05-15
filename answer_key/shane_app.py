import sys
sys.path.append("../")

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv(), override=True)

import os
import uuid
from time import sleep
from loguru import logger
from typing import Generator, Any
import streamlit as st
from dotenv import load_dotenv

from tiktoken import Encoding, get_encoding
from src.llm.prompt_templates import (
    huberman_system_message,
    question_answering_prompt_series,
    generate_prompt_series
)
# from src.backend.reader.prompt_templates import (
#     odin_system_message,
#     json_question_answering_series,
#     context_block,
# )
from src.llm.llm_utils import load_azure_openai
from app_functions import validate_token_threshold
from src.reranker import ReRanker
from src.database.weaviate_interface_v4 import WeaviateWCS
from src.database.database_utils import get_weaviate_client
from src.conversation import Conversation, Message

abs_path = os.path.abspath("../../")
sys.path.append(abs_path)

## PAGE CONFIGURATION
st.set_page_config(
    page_title="Huberman Lab",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="auto",
    menu_items=None,
)
collection_name = 'Huberman_minilm_256'
turbo = 'gpt-3.5-turbo-0125'
claude = 'claude-3-haiku-20240307'
anthro_api_key = os.getenv('ANTHROPIC_API_KEY')
data_path = '../data/huberman_labs.json'
# content_data_path = '../impact-theory-newft-256.parquet'
CONVERSATION_KEY = "conversation"
MESSAGE_BUILDER_KEY = "message_builder"
EMBEDDING_MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2" #"BAAI/bge-base-en"
INDEX_NAME = "odin-bge-768-feb5"
TITLES = "titles"
PAGES = "pages"
huberman_icon = "./assets/huberman_logo.png"
uplimit_icon = './assets/uplimit_logo.jpg'


## RETRIEVER
@st.cache_resource
def get_retriever() -> WeaviateWCS:
    return get_weaviate_client(model_name_or_path=EMBEDDING_MODEL_PATH)

retriever = get_retriever()

## Display fields
LLM_CONTENT_FIELD = "content"
DSIPLAY_CONTENT_FIELD = "content"
app_display_fields = retriever.return_properties 
logger.info(app_display_fields)


## READER
# llm_api_key = os.environ["AZURE_OPENAI_API_KEY"]
# llm_api_version = os.environ["AZURE_OPENAI_API_VERSION"]
# llm_api_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
# llm_names = {
#     "short_context": "odin-chat-model",
#     "long_context": "odin-chat-16k",
#     "json_mode": "odin-chat-gpt4",
# }


# Cache reranker
@st.cache_resource
def get_reranker() -> ReRanker:
    return ReRanker()


# Cache LLM client
# @st.cache_resource
# def get_llm_client(api_key, api_version, azure_endpoint):
#     return AzureOpenAI(
#         api_key=api_key, api_version=api_version, azure_endpoint=azure_endpoint
#     )


# Cache LLM
@st.cache_resource
def get_llm(model_name: str='gpt-35-turbo'):
    return load_azure_openai(model_name=model_name)


## TOKENIZER
@st.cache_resource
def get_encoding_model(model_name) -> Encoding:
    return get_encoding(model_name)


reranker = get_reranker()
# aoai_client = get_llm_client(llm_api_key, llm_api_version, llm_api_endpoint)
# llm = get_llm(_client=aoai_client)
llm = get_llm()
encoding = get_encoding_model("cl100k_base")


# Run the chat interface within Streamlit
def run_chat_interface():
    """Run the chat interface."""
    create_chat_area(st.session_state[CONVERSATION_KEY])

    # Chat controls
    clear_button = st.button("Clear Chat History")
    user_input = st.chat_input("Ask something:")

    # Clear chat history
    if clear_button:
        st.session_state[CONVERSATION_KEY] = get_new_conversation()
        st.rerun()

    # Handle user input and generate assistant response
    if user_input or st.session_state.streaming:
        with st.chat_message("user", avatar=uplimit_icon):
            st.write(user_input)
        process_user_input(user_input)
        st.rerun()


def create_chat_area(conversation: Conversation):
    """Display the chat history."""
    for msg in conversation.queue_to_list():
        # Use this call to pass in the Odin icon
        if msg["role"] == "assistant":
            with st.chat_message(name="assistant", avatar=huberman_icon):
                st.write(msg["content"])
        else:
            with st.chat_message(msg["role"], avatar=uplimit_icon):
                st.write(msg["content"])


def get_new_conversation(user: str = "user"):
    """Create a new conversation."""
    new_convo_id = user + "-" + str(uuid.uuid4())
    return Conversation(
        conversation_id=new_convo_id,
        system_message=Message(role="assistant", content=huberman_system_message),
    )


def process_user_input(user_input):
    """Process the user input and generate the assistant response."""
    if user_input:
        # 1. Run rag search
        results = retriever.hybrid_search(
            user_input, collection_name=collection_name, return_properties=app_display_fields, limit=200
        )

        # 2. Rerank search results using semantic reranker
        reranked_results = reranker.rerank(
            results, user_input, apply_sigmoid=False, top_k=5
        )

        # 3. Validate token threshold
        valid_results = validate_token_threshold(
            reranked_results,
            question_answering_prompt_series,
            query=user_input,
            tokenizer=encoding,
            token_threshold=6000,
            content_field=LLM_CONTENT_FIELD
        )

        # 4. Generate context series
        context_series = generate_prompt_series(user_input, valid_results, 1)

        # 5. Generate gpt4_json_question_prompt
        # assistant_message_answering_series = question_answering_prompt_series.format(
        #     question=user_input, context=context_series
        # )
        
        # Add user_query to the conversation
        st.session_state[CONVERSATION_KEY].add_message(
            Message(role="user", content=user_input)
        )
        logger.info(st.session_state[CONVERSATION_KEY].queue_to_list())
        # 6. Generate assistant response
        with st.chat_message(name="assistant", avatar=huberman_icon):
            gpt_answer = st.write_stream(
                chat(
                    user_message=context_series,
                    max_tokens=1000
                )
            )

        ref_docs = list(
            set(list(zip(st.session_state[TITLES], st.session_state[PAGES])))
        )
        if any(ref_docs):
            with st.expander("Reference Documents", expanded=False):
                for i, doc in enumerate(ref_docs, start=1):
                    st.markdown(f"{i}. **{doc[0]}**: &nbsp; &nbsp; page {doc[1]}")
            st.session_state[TITLES], st.session_state[PAGES] = [], []

        # 7. Add assistant response to the conversation
        st.session_state[CONVERSATION_KEY].add_message(
            Message(role="assistant", content=gpt_answer)
        )
        #     st.session_state.generator = gpt_answer
        #     st.session_state.streaming = True
        #     st.rerun()
        # else:
        #     update_assistant_response()


# Generate chat responses using the OpenAI API
def chat(
    user_message: str,
    max_tokens: int=250,
    temperature: float=0.5,
 ) -> Generator[Any, Any, None]:
    """Generate chat responses using an LLM API.
    Stream response out to UI.
    """
    completion = llm.chat_completion(huberman_system_message,
                                     user_message=user_message,
                                     temperature=temperature,
                                     max_tokens=max_tokens,
                                     stream=True)

    colon_count = 0
    full_json = []
    double_quote_count = 0
    double_quote = '"'
    colon = ":"

    for chunk in completion:
        sleep(0.05)
        if any(chunk.choices):
            content = chunk.choices[0].delta.content
            if content:
                full_json.append(content)
                yield content

    # for chunk in completion:
    #     sleep(0.05)
    #     if any(chunk.choices):
    #         content = chunk.choices[0].delta.content
    #         if content:
    #             full_json.append(content)
    #             if colon in content:
    #                 colon_count += 1
    #                 continue
    #         if colon_count >= 1:
    #             if double_quote_count < 2:
    #                 yield content
    #             if content and double_quote in content:
    #                 double_quote_count += 1

    answer = "".join(full_json)
    # logger.info(answer)

    # json_response = json.loads("".join(full_json))
    # logger.info(json_response)
    # answer, title, page = parse_json_response(json_response)
    # st.session_state[TITLES].extend(title)
    # st.session_state[PAGES].extend(page)
    st.session_state[MESSAGE_BUILDER_KEY] = {"role": "assistant", "content": answer}

# Main function to run the Streamlit app
def main():
    """Main function to run the Streamlit app."""
    st.markdown(
        """<style>.block-container{max-width: 66rem !important;}</style>""",
        unsafe_allow_html=True,
    )
    st.title("Chat with the Huberman Lab podcast")
    st.markdown("---")

    # Session state initialization
    if CONVERSATION_KEY not in st.session_state:
        st.session_state[CONVERSATION_KEY] = get_new_conversation()
    if "streaming" not in st.session_state:
        st.session_state.streaming = False
    if TITLES not in st.session_state:
        st.session_state[TITLES] = []
    if PAGES not in st.session_state:
        st.session_state[PAGES] = []

    run_chat_interface()

if __name__ == "__main__":
    main()
