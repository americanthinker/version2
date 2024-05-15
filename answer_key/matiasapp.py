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
collection_name = 'Huberman_minilm_256'
turbo = 'gpt-3.5-turbo-0125'
claude = 'claude-3-haiku-20240307'
anthro_api_key = os.getenv('ANTHROPIC_API_KEY')
data_path = '../data/huberman_labs.json'
embedding_model_path = 'sentence-transformers/all-MiniLM-L6-v2'  # './models/finetuned-all-MiniLM-L6-v2-300/'
###################################

## RETRIEVER
retriever = get_weaviate_client(model_name_or_path=embedding_model_path)
if retriever._client.is_live():
    logger.info('Weaviate is ready!')

## RERANKER
reranker = ReRanker()

## QA MODEL
llm = LLM(turbo)

## TOKENIZER
encoding = get_encoding("cl100k_base")

## Display properties
display_properties = retriever.return_properties + ['summary', 'length_seconds', 'thumbnail_url', 'episode_url']

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
        alpha_input = st.slider('Alpha for Hybrid Search', 0.00, 1.00, 0.40, step=0.05)
        retrieval_limit = st.slider('Hybrid Search Retrieval Results', 10, 300, 10, step=10)
        reranker_topk = st.slider('Reranker Top K', 1, 5, 3, step=1)
        temperature_input = st.slider('Temperature of LLM', 0.0, 2.0, 0.10, step=0.10)
        verbosity = st.slider('LLM Verbosity', 0, 2, 0, step=1)
    
    # if collection_name == 'Ada_data_256':
    #     client = WeaviateClient(api_key, url, model_name_or_path='text-embedding-ada-002',openai_api_key=os.environ['OPENAI_API_KEY'])
        
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
        hybrid_response = retriever.hybrid_search(query, 
                                                  collection_name, 
                                                  alpha=alpha_input, 
                                                  return_properties=display_properties,
                                                  filter=guest_filter,
                                                  limit=retrieval_limit)
        # rerank results
        ranked_response = reranker.rerank(  hybrid_response, 
                                            query, 
                                            apply_sigmoid=True, 
                                            top_k=reranker_topk)
        # expanded_response = expand_content(ranked_response, cache, content_key='doc_id', create_new_list=True)
        
        # validate token count is below threshold
        token_threshold = 8000 #if model_name == model_ids[0] else 3500
        content_field = 'content'
        valid_response = validate_token_threshold(  ranked_response, 
                                                    question_answering_prompt_series, 
                                                    query=query,
                                                    tokenizer=encoding,# variable from ENCODING,
                                                    token_threshold=token_threshold, 
                                                    content_field=content_field,
                                                    verbose=True)
        
        make_llm_call = True
        # prep for streaming response
        st.subheader("Response from Impact Theory (context)")
        with st.spinner('Generating Response...'):
            st.markdown("----")
            #creates container for LLM response
            chat_container, response_box = [], st.empty()
                
            ##############
            # START CODE #
            ##############
            # generate LLM prompt
            prompt = generate_prompt_series(query=query, results=valid_response, verbosity_level=verbosity)
            json_mode = False
            if make_llm_call:
                with st.chat_message('Huberman Labs', avatar='./assets/huberman_logo.png'):
                    if json_mode:
                        st.write_stream(stream_json_chat(llm, prompt))
                    else:
                        stream_obj = stream_chat(llm, prompt, temperature=temperature_input)
                        st.write_stream(stream_obj)
            string_completion = ' '.join([c for c in stream_obj])
            call_cost = completion_cost(completion=string_completion, 
                                        model=turbo, 
                                        prompt=huberman_system_message + ' ' + prompt,
                                        call_type='completion')
            st.session_state['cost_counter'] += call_cost
            logger.info(f'TOTAL SESSION COST: {st.session_state["cost_counter"]}')

                # try: 
                #     for resp in llm.chat_completion(system_message=huberman_system_message,
                #                                     user_message=prompt,
                #                                     temperature=temperature_input,
                #                                     max_tokens=350,
                #                                     stream=True):                
                # ##############
                # #  END CODE  #
                # ##############
                #         #inserts chat stream from LLM
                #         with response_box:
                #             content = resp.choices[0].delta.content
                #             if content:
                #                 chat_container.append(content)
                #                 result = "".join(chat_container).strip()
                #                 st.write(f'{result}')
                # except Exception as e:
                #     print(e)
                # except BadRequestError: 
                #     logger.info('Making request with smaller context...')
                #     valid_response = validate_token_threshold(ranked_response, 
                #                                                 question_answering_prompt_series, 
                #                                                 query=query,
                #                                                 tokenizer=encoding,# variable from ENCODING,
                #                                                 token_threshold=token_threshold - 500, 
                #                                                 verbose=True)
                #     prompt = generate_prompt_series(query=query, results=valid_response)
                #     for resp in llm.get_chat_completion(prompt=prompt,
                #                                 temperature=temperature_input,
                #                                 max_tokens=350,
                #                                 show_response=True,
                #                                 stream=True):                
                #         try:
                #             #inserts chat stream from LLM
                #             with response_box:
                #                 content = resp.choices[0].delta.content
                #                 if content:
                #                     chat_container.append(content)
                #                     result = "".join(chat_container).strip()
                #                     st.write(f'{result}')
                #         except Exception as e:
                #             print(e)
            
            ##############
            # START CODE #
            ##############
            st.subheader("Search Results")
            for i, hit in enumerate(valid_response):
                col1, col2 = st.columns([7, 3], gap='large')
                episode_url = hit['episode_url']
                title = hit['title']
                show_length = hit['length_seconds']
                time_string = convert_seconds(show_length) # convert show_length to readable time string
            # ##############
            # #  END CODE  #
            # ##############
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