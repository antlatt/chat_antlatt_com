import streamlit as st
from langchain.llms.ollama import Ollama
from langchain.chat_models import ChatOllama
import langchain.document_loaders
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.gpt4all import GPT4AllEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain.schema import messages_from_dict, messages_to_dict
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, ConversationChain, LLMChain
from langchain.prompts import PromptTemplate
import json
import os

st.set_page_config(page_title="chat.antlatt.com", page_icon=None, layout="centered", initial_sidebar_state="collapsed")
ollama = ChatOllama(base_url='http://192.168.1.113:11434', model='mistral', temperature=0.1, streaming=True)
persist_directory = "./vectorstores/db/"
#set_llm_cache(InMemoryCache())


### CREATE VECTORSTORE FUNCTION

def db_lookup():
    try:
        if url is not None:
            loader = langchain.document_loaders.WebBaseLoader(url)

            documents = loader.load()

            len(documents)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

            texts = text_splitter.split_documents(documents)

            len(texts)

            persist_directory = "./vectorstores/db/"

            embeddings = GPT4AllEmbeddings()

        
            vectordb = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=persist_directory)

            vectordb.persist()
            vectordb = None
    except:
        if dir is not None:
            pdf_folder_path = './pdfs/'
            dir_loader = PyPDFDirectoryLoader(pdf_folder_path)
            dir_docs = dir_loader.load()
            len(dir_docs)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            dir_texts = text_splitter.split_documents(dir_docs)
            len(dir_texts)
            persist_directory = "./vectorstores/db/"
            embeddings = GPT4AllEmbeddings()
            dir_vectordb = Chroma.from_documents(documents=dir_texts, embedding=embeddings, persist_directory=persist_directory)
            dir_vectordb.persist()
            dir_vectordb = None
    finally:
        if pdf is not None:
            pdf_reader = PdfReader(pdf)
    #        st.write(pdf_reader)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
    #        st.write(text)

    #        len(pdf_documents)
        
            pdf_text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    #        st.write(pdf_text_splitter)

            pdf_texts = pdf_text_splitter.split_text(text=text)

            len(pdf_texts)

    #        st.write(pdf_splits)

            persist_directory = "./vectorstores/db/"

            pdf_embeddings = GPT4AllEmbeddings()
        
            pdf_vectordb = Chroma.from_texts(pdf_texts, embedding=pdf_embeddings, persist_directory=persist_directory)

            pdf_vectordb.persist()
            pdf_vectordb = None


    
# Sidebar Contents

with st.sidebar:
    st.sidebar.title('ANTLATT.com')
    st.sidebar.header('Add More Data to the Database')

##SIDEBAR PDF INPUT    
    pdf = st.sidebar.file_uploader("Upload a PDF", type="pdf", disabled=False)

##SIDEBAR DIR INPUT##
    dir = st.selectbox('load stored pdf directory?', ('Yes', 'No', 'None'), index=None)
    st.write('You selected:', dir)

###SIDEBAR URL INPUT                
    url = st.sidebar.text_input('Enter a URL', placeholder="enter url here", disabled=False)
    with st.form('myform2', clear_on_submit=True):
        
        persist_directory = "./vectorstores/db"

        submitted = st.form_submit_button('Submit', disabled=not(url or dir or pdf))
        if submitted:
            with st.spinner('Creating VectorStore, Saving to Disk...'):
                db_lookup()
                with st.success('Done!'):
                    st.write('VectorStore Created and Saved to Disk')

    
    st.markdown('''
    ## About
    This is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io)
    - [Langchain](https://python.langchain.com)
    - [Ollama](https://ollama.com)
    - [Neural-Chat](https://huggingface.co/illuin/mistral-7b)

    ''')
    add_vertical_space(5)
    st.write('Made by [antlatt](https://www.antlatt.com)')


### DB VECTOR LOOKUP FUNCTION

#def db_vector_lookup(question):
#    if question is not None:
#        persist_directory = "./vectorstores/db/"
#        embeddings = GPT4AllEmbeddings()
#        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

#        retriever = vectordb.as_retriever()
#        docs = retriever.get_relevant_documents(question)
#        len(docs)
#        retriever = vectordb.as_retriever(search_kwags={"k": 3})
#        retriever.search_type = "similarity"
#        retriever.search_kwargs = {"k": 3}
#        qachain = RetrievalQA.from_chain_type(ollama, chain_type="stuff", retriever=retriever, return_source_documents=True)

#        return qachain({"query": question})

st.title('ANTLATT.com')
st.header('Chat with Your Documents')
st.write("PDF's in Current Database: ", os.listdir('./pdfs/'))
if dir:
    st.write("Current Database: ", os.listdir('./pdfs/'))
if pdf:
    st.write("PDF database currently loaded: ", pdf.name)

###Query database with url or pdf
#question = st.text_input('Enter your question:', placeholder = 'enter a question here.', disabled=False)

#result = []
#with st.form('myform3', clear_on_submit=True):
#    persist_directory = "/vectorstores/db/"

#    submitted = st.form_submit_button('Submit', disabled=not(question))
#    if submitted:
#        with st.spinner('Operation In Progress...'):
#            response = db_vector_lookup(question)
#            result.append(response)


#if len(result):
#    with st.container():
#       st.write(response)


### Chat App
#user_prompt = st.chat_input('Enter your message:', key="user_prompt")
#if user_prompt:
#    st.write(f'You: {user_prompt}'
#)

# Initialize chat history
#if "messages" not in st.session_state:
#    st.session_state.messages = []

# Display chat messages from history on app rerun
#for message in st.session_state.messages:
#    with st.chat_message(message["role"]):
#        st.markdown(message["content"])

# React to user input
#if prompt := st.chat_input("Send a Chat Message to the AI Assistant"):
#    st.session_state.messages.append({"role": "user", "content": prompt})
#    with st.chat_message("user"):
#        st.markdown(prompt)

#    with st.chat_message("assistant"):
#        message_placeholder = st.empty()
#        full_response = ollama.predict(prompt)
#        message_placeholder.markdown(full_response)
#    st.session_state.messages.append({"role": "assistant", "content": full_response})

### NEW CHAT APP ####
history = StreamlitChatMessageHistory(key="chat_messages")

msgs = StreamlitChatMessageHistory(key="special_app_key")

memory = ConversationBufferMemory(memory_key="history", chat_memory=msgs)
if len(msgs.messages) == 0:
    msgs.add_ai_message("Hello, how can I help you today?")

template = """You are a friendly and helpful AI chatbot named Jarvis having a conversation with a human. Give extremely detailed answers to the human's questions using the context provided. 
{history}
Human: {human_input}
AI: """
prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)

llm_chain = LLMChain(llm=ollama, prompt=prompt, verbose=True, memory=memory)

#conversation = ConversationalRetrievalChain.from_llm(llm=ollama, retriever=retriever, memory=memory)

for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

if prompt := st.chat_input():
    st.chat_message("human").write(prompt)
    persist_directory = "./vectorstores/db/"
    embeddings = GPT4AllEmbeddings()
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = vectordb.as_retriever()
    docs = retriever.get_relevant_documents(prompt)
    len(docs)
    retriever = vectordb.as_retriever(search_kwags={"k": 5})
    retriever.search_type = "similarity"
    retriever.search_kwargs = {"k": 5}
#    conversation = ConversationalRetrievalChain.from_llm(llm=ollama, retriever=retriever, verbose=True, memory=memory)
#    qachain = RetrievalQA.from_chain_type(ollama, chain_type="stuff", retriever=retriever, return_source_documents=False, verbose=True, memory=memory)
#    lookup = qachain({"query": prompt})

    response = llm_chain.run(prompt)
    st.chat_message("ai").write(response)

### Chat App End

#if __name__ == "__main__":
#    main()
