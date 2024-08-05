from dataclasses import dataclass
from typing import Literal
import streamlit as st
import os
# from llamaapi import LlamaAPI
# from langchain_experimental.llms import ChatLlamaAPI
from langchain.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import streamlit.components.v1 as components
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
import time
import os
os.environ['PINECONE_API_KEY'] = <PINECONE API>
pc = Pinecone()

HUGGINGFACEHUB_API_TOKEN = st.secrets['HUGGINGFACEHUB_API_TOKEN']
from langchain.vectorstores import Pinecone
@dataclass
class Message:
    """Class for keeping track of a chat message."""
    origin: Literal["👤 Human", "👨🏻‍⚖️ Ai"]
    message: str


def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings


def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "conversation" not in st.session_state:
        # llama = LlamaAPI(st.secrets["LlamaAPI"])
        # model = ChatLlamaAPI(client=llama)
        chat = ChatGroq(temperature=0.5, groq_api_key=st.secrets["Groq_api"], model_name="llama3-70b-8192")

        embeddings = download_hugging_face_embeddings()

        
        index_name = "medical-advisor"

        if index_name in pc.list_indexes().names():
            print("index already exists" , index_name)
            index= pc.Index(index_name) #your index which is already existing and is ready to use
            print(index.describe_index_stats())
          # put in the name of your pinecone index here

        docsearch = Pinecone.from_existing_index(index_name, embeddings)

        prompt_template = """
            You are a trained bot to guide people about their medical concerns acting as Doctor. You will answer user's query with your knowledge and the context provided. 
            
            If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
            Provide very detailed answer as Doctor.
            
            Use the following pieces of context to answer the users question.
            Context: {context}
            Question: {question}
            Only return the helpful answer below and nothing else.
            Helpful answer:
            """

        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        
        #chain_type_kwargs = {"prompt": PROMPT}
        message_history = ChatMessageHistory()
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            chat_memory=message_history,
            return_messages=True,
            )
        retrieval_chain = ConversationalRetrievalChain.from_llm(llm=chat,
                                                      chain_type="stuff",
                                                      retriever=docsearch.as_retriever(
                                                          search_kwargs={'k': 3}),
                                                      return_source_documents=True,
                                                      combine_docs_chain_kwargs={"prompt": PROMPT},
                                                      memory= memory
                                                     )

        st.session_state.conversation = retrieval_chain


def on_click_callback():
    human_prompt = st.session_state.human_prompt
    st.session_state.human_prompt=""
    response = st.session_state.conversation(
        human_prompt
    )
    llm_response = response['answer']
    st.session_state.history.append(
        Message("👤 Human", human_prompt)
    )
    st.session_state.history.append(
        Message("👨🏻‍⚖️ Ai", llm_response)
    )


initialize_session_state()

st.title("Medical Advisor Chatbot 🇮🇳")



chat_placeholder = st.container()
prompt_placeholder = st.form("chat-form")

with chat_placeholder:
    for chat in st.session_state.history:
        st.markdown(f"{chat.origin} : {chat.message}")

with prompt_placeholder:
    st.markdown("**Chat**")
    cols = st.columns((6, 1))
    cols[0].text_input(
        "Chat",
        label_visibility="collapsed",
        key="human_prompt",
    )
    cols[1].form_submit_button(
        "Submit",
        type="primary",
        on_click=on_click_callback,
    )
