import streamlit as st
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.callbacks import get_openai_callback
from langchain.memory import ConversationBufferMemory
from langchain.memory import StreamlitChatMessageHistory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, AIMessage
import tiktoken
import json
import base64

def main():
    st.set_page_config(page_title="kangsinchat", page_icon="🏫")
    st.image('energy.png')
    st.title("_:red[에너지 학습 도우미 ]_ 🏫")
    st.header("😶주의!이 챗봇은 참고용으로 사용하세요!", divider='rainbow')

    if "conversation" not in st.session_state:
        clear_button = st.button("대화 내용 삭제", key="clear_button")
        if clear_button:
            st.session_state.chat_history = []
            st.session_state.messages = [{"role": "assistant", "content": "에너지 학습에 대해 물어보세요!😊"}]
            st.experimental_rerun()  # 화면을 다시 로드하여 대화 내용을 초기화

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant",                                          
                                         "content": "에너지 학습에 대해 물어보세요!😊"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("생각 중..."):
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(response)
