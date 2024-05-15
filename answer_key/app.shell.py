from tiktoken import get_encoding
from src.database.weaviate_interface_v4 import WeaviateWCS
from src.llm.llm_interface import LLM
from src.llm.prompt_templates import question_answering_prompt_series, huberman_system_message
from app_functions import (convert_seconds, search_result,
                          validate_token_threshold, load_data)
from src.reranker import ReRanker
from loguru import logger 
import streamlit as st
import sys
import json
import os

# load environment variables
from dotenv import load_dotenv, find_dotenv
env_path = find_dotenv(raise_error_if_not_found=True)
load_dotenv(env_path, override=True)


## PAGE CONFIGURATION
st.set_page_config(page_title="Impact Theory", 
                   page_icon=None, 
                   layout="wide", 
                   initial_sidebar_state="auto", 
                   menu_items=None)
##############
# START CODE #
##############
data_path = './data/huberman_labs.json'
## RETRIEVER

## RERANKER

## LLM 

## ENCODING

## INDEX NAME

##############
#  END CODE  #
##############
data = load_data(data_path)
#creates list of guests for sidebar
guest_list = sorted(list(set([d['guest'] for d in data])))

def main():
        
    with st.sidebar:
        guest = st.selectbox('Select Guest', options=guest_list, index=None, placeholder='Select Guest')

    st.image('./assets/impact-theory-logo.png', width=400)
    st.subheader(f"Chat with the Impact Theory podcast: ")
    st.write('\n')
    col1, _ = st.columns([7,3])
    with col1:
        query = st.text_input('Enter your question: ')
        st.write('\n\n\n\n\n')

        if query:
            ##############
            # START CODE #
            ##############

            st.write('Hmmm...this app does not seem to be working yet.  Please check back later.')
            if guest:
                st.write(f'However, it looks like you selected {guest} as a filter.')
            # make hybrid call to weaviate
            hybrid_response = None
            # rerank results
            ranked_response = None
            # validate token count is below threshold
            # valid_response = validate_token_threshold(ranked_response, 
                                                    #    question_answering_prompt_series, 
                                                    #    query=query,
                                                    #    tokenizer= # variable from ENCODING,
                                                    #    token_threshold=4000, 
                                                    #    verbose=True)
            ##############
            #  END CODE  #
            ##############

            # # generate LLM prompt
            # prompt = generate_prompt_series(base_prompt=question_answering_prompt_series, query=query, results=valid_response)
            
            # # prep for streaming response
            # st.subheader("Response from Impact Theory (context)")
            # with st.spinner('Generating Response...'):
            #     st.markdown("----")
            #     #creates container for LLM response
            #     chat_container, response_box = [], st.empty()
            #     
            #     # execute chat call to LLM
            #                  ##############
            #                  # START CODE #
            #                  ##############
            #     

            #                  ##############
            #                  #  END CODE  #
            #                  ##############
            #         try:
                          #inserts chat stream from LLM
            #             with response_box:
                        #     content = resp.choices[0].delta.content
                        #     if content:
                        #         chat_container.append(content)
                        #         result = "".join(chat_container).strip()
                        #         st.write(f'{result}')
                        # except Exception as e:
                        #     print(e)
                        #     continue
            # ##############
            # # START CODE #
            # ##############
            # st.subheader("Search Results")
            # for i, hit in enumerate(valid_response):
            #     col1, col2 = st.columns([7, 3], gap='large')
            #     image = # get thumbnail_url
            #     episode_url = # get episode_url
            #     title = # get title
            #     show_length = # get length
            #     time_string = # convert show_length to readable time string
            # ##############
            # #  END CODE  #
            # ##############
            #     with col1:
            #         st.write( search_result(  i=i, 
                                                # url=episode_url,
                                                # guest=hit['guest'],
                                                # title=title,
                                                # content=hit['content'], 
                                                # length=time_string),
            #                 unsafe_allow_html=True)
            #         st.write('\n\n')
            #     with col2:
            #         # st.write(f"<a href={episode_url} <img src={image} width='200'></a>", 
            #         #             unsafe_allow_html=True)
            #         st.image(image, caption=title.split('|')[0], width=200, use_column_width=False)

if __name__ == '__main__':
    main()