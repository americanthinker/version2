import os
import sys
sys.path.append('../')
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

from tiktoken import get_encoding
from weaviate.classes.query import Filter
from litellm import completion_cost
from loguru import logger 
import streamlit as st

from src.database.weaviate_interface_v4 import WeaviateWCS
from src.database.database_utils import get_weaviate_client
from src.llm.llm_interface import LLM
from src.reranker import ReRanker
from src.llm.prompt_templates import generate_prompt_series, huberman_system_message, question_answering_prompt_series
from app_functions import (convert_seconds, search_result, validate_token_threshold,
                           stream_chat, stream_json_chat, load_data)

 
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
collection_name = None
aoai_model = 'gpt-3.5-turbo-0125'
claude = 'claude-3-haiku-20240307'
data_path = '../data/huberman_labs.json'
embedding_model_path = 'put your fine-tuned model here'
###################################

## RETRIEVER
retriever = None
if retriever._client.is_live():
    logger.info('Weaviate is ready!')

## RERANKER
reranker = None

## QA MODEL
llm = None

## TOKENIZER
encoding = get_encoding("cl100k_base")

## Display properties
display_properties = None

## Data
data = load_data(data_path)
#creates list of guests for sidebar
guest_list = sorted(list(set([d['guest'] for d in data])))
available_collections = ['Huberman_minilm_128', 'Huberman_minilm_256', 'Huberman_minilm_512']

## COST COUNTER
if not st.session_state.get('cost_counter'):
    st.session_state['cost_counter'] = 0

def main(retriever: WeaviateWCS):
    #################
    #### SIDEBAR ####
    #################
    with st.sidebar:
        # filter_guest_checkbox = st.checkbox('Filter Guest')
        collection_name = st.selectbox( 'Collection Name:',options=available_collections, index=None,placeholder='Select Collection Name')
        guest_input = st.selectbox('Select Guest', options=guest_list,index=None, placeholder='Select Guest')
        alpha_input = None
        retrieval_limit = None
        reranker_topk = None
        temperature_input = None
        verbosity = None
        
    # retriever.return_properties.append('expanded_content')

    ##############################
    ##### SETUP MAIN DISPLAY #####
    ##############################
    st.image('./assets/hlabs_logo.png', width=400)
    st.subheader("Search with the Huberman Lab podcast:")
    st.write('\n')
    col1, _ = st.columns([7,3])
    with col1:
        query = st.text_input('Enter your question: ')
        st.write('\n\n\n\n\n')

    ########################
    ##### SEARCH + LLM #####
    ########################
    if query and not collection_name:
        raise ValueError('Please first select a collection name')
    if query:
        # make hybrid call to weaviate
        guest_filter = Filter.by_property(name='guest').equal(guest_input) if guest_input else None
        hybrid_response = None
        # rerank results
        ranked_response = None        
        # validate token count is below threshold
        token_threshold = 8000
        content_field = 'content'
        valid_response = validate_token_threshold(  ranked_response, 
                                                    question_answering_prompt_series, 
                                                    query=query,
                                                    tokenizer=encoding,
                                                    token_threshold=token_threshold, 
                                                    content_field=content_field,
                                                    verbose=True)
        #set to False to skip LLM call
        make_llm_call = True
        # prep for streaming response
        st.subheader("Response from Impact Theory (context)")
        with st.spinner('Generating Response...'):
            st.markdown("----")                
            # generate LLM prompt
            prompt = generate_prompt_series(query=query, results=valid_response, verbosity_level=verbosity)
            if make_llm_call:
                with st.chat_message('Huberman Labs', avatar='./assets/huberman_logo.png'):
                    stream_obj = stream_chat(llm, prompt, temperature=temperature_input)
                    st.write_stream(stream_obj)
            string_completion = ' '.join([c for c in stream_obj])
            call_cost = completion_cost(completion=string_completion, 
                                        model=turbo, 
                                        prompt=huberman_system_message + ' ' + prompt,
                                        call_type='completion')
            st.session_state['cost_counter'] += call_cost
            logger.info(f'TOTAL SESSION COST: {st.session_state["cost_counter"]}')

    ##################
    # SEARCH DISPLAY #
    ##################
            st.subheader("Search Results")
            for i, hit in enumerate(valid_response):
                col1, col2 = st.columns([7, 3], gap='large')
                episode_url = hit['episode_url']
                title = hit['title']
                show_length = hit['length_seconds']
                time_string = convert_seconds(show_length) # convert show_length to readable time string
                with col1:
                    st.write( search_result(i=i, 
                                            url=episode_url,
                                            guest=hit['guest'],
                                            title=title,
                                            content=ranked_response[i]['content'], 
                                            length=time_string),
                                            unsafe_allow_html=True)
                    st.write('\n\n')

                with col2:
                    image = hit['thumbnail_url']
                    st.image(image, caption=title.split('|')[0], width=200, use_column_width=False)
                    st.markdown(f'<p style="text-align": right;"><b>Guest: {hit["guest"]}</b>', unsafe_allow_html=True)

if __name__ == '__main__':
    main(retriever)