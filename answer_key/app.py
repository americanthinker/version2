import sys
sys.path.append('../')
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

from tiktoken import get_encoding
from weaviate.classes.query import Filter
from src.database.database_utils import get_weaviate_client
from src.llm.llm_interface import LLM
from src.llm.prompt_templates import generate_prompt_series
from app_functions import (convert_seconds, search_result,
                           stream_chat, stream_json_chat, load_data)
from src.reranker import ReRanker
from loguru import logger 
import streamlit as st
import os

## PAGE CONFIGURATION
st.set_page_config(page_title="Huberman Labs", 
                   page_icon=None, 
                   layout="wide", 
                   initial_sidebar_state="auto", 
                   menu_items=None)

###################################
#### CLASS NAME AND DATA PATHS ####
#### WILL CHANGE USER TO USER  ####
###################################
collection_name = 'Huberman_minilm_256'
turbo = 'gpt-3.5-turbo-0125'
claude = 'claude-3-haiku-20240307'
anthro_api_key = os.getenv('ANTHROPIC_API_KEY')
data_path = '../data/huberman_labs.json'
embedding_model_path = 'sentence-transformers/all-MiniLM-L6-v2'  # './models/finetuned-all-MiniLM-L6-v2-300/'
###################################

## RETRIEVER
client = get_weaviate_client(model_name_or_path=embedding_model_path)
if client._client.is_live():
    logger.info('Weaviate is ready!')
## RERANKER
reranker = ReRanker()
## QA MODEL
llm = LLM(turbo)
## TOKENIZER
encoding = get_encoding("cl100k_base")
## Display properties
display_properties = client.return_properties + ['summary', 'length_seconds', 'thumbnail_url', 'episode_url']
#loads cache of content data
# content_cache = load_content_cache(content_data_path)
# #loads data for property extraction
data = load_data(data_path)
#creates list of guests for sidebar
guest_list = sorted(list(set([d['guest'] for d in data])))

def main():
    
    with st.sidebar:
        guest = st.selectbox('Select Guest', options=guest_list, index=None, placeholder='Select Guest')

    subheader_entry = 'Huberman Lab podcast'
    st.image('./assets/hlabs_logo.png', width=400)
    st.subheader(f"Chat with the {subheader_entry}: ")

    st.write('\n')
    col1, _ = st.columns([7,3])
    with col1:
        query = st.text_input('Enter your question: ')
        st.write('\n\n\n\n\n')
    if query:
        with st.spinner('Searching...'):
            filter = Filter.by_property('guest').equal(guest) if guest else None
            hybrid_response = client.hybrid_search(request=query,
                                                   collection_name=collection_name,
                                                   query_properties=['content', 'summary', 'title', 'guest'],
                                                   alpha=0.3,
                                                   limit=200,
                                                   filter=filter,
                                                   return_properties=display_properties
                                                   )
            ranked_response = reranker.rerank(hybrid_response, query, top_k=3)
            
            # valid_response = validate_token_threshold(ranked_response, 
            #                                           question_answering_prompt_series, 
            #                                           query=query,
            #                                           tokenizer=encoding,
            #                                           token_threshold=6000, 
            #                                           verbose=True)
        json_mode = False
        user_message = generate_prompt_series(query, ranked_response, 2)
        if json_mode:
            user_message += '\nReturn your response in JSON format, using the word "answer" as the key. Also include the show guest using the "guest" key.'
            
        #execute chat call to OpenAI
        # st.subheader("Response from Huberman Labs")
        with st.spinner('Generating Response...'):
            st.markdown("----")
            with st.chat_message('Huberman Labs', avatar='./assets/huberman_logo.png'):
                if json_mode:
                    st.write_stream(stream_json_chat(llm, user_message))
                else:
                    st.write_stream(stream_chat(llm, user_message))
        st.markdown("----")
        st.subheader("Search Results")
        for i, hit in enumerate(ranked_response):
            col1, col2 = st.columns([7, 3], gap='large')
            episode_url = hit['episode_url']
            title = hit['title']
            show_length = hit['length_seconds']
            time_string = convert_seconds(show_length)

            #break out search reults into two columns
            #column 1 = search result
            with col1:
                st.write(search_result( i=i, 
                                        url=episode_url,
                                        guest=hit['guest'],
                                        title=title,
                                        content=hit['content'],
                                        length=time_string),
                        unsafe_allow_html=True)
                st.write('\n\n')

            #column 2 = thumbnail image
            with col2:
                image = hit['thumbnail_url']
                st.image(image, caption=title.split('|')[0], width=200, use_column_width=False)
                st.markdown(f'<p style="text-align": right;"><b>Guest: {hit["guest"]}</b>', unsafe_allow_html=True)
           
if __name__ == '__main__':
    main()