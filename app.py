from io import BytesIO
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

def extract_text_from_pdf(pdf_file_obj):
    pdf_reader = PdfReader(BytesIO(pdf_file_obj.getbuffer()))
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page_obj = pdf_reader.pages[page_num]
        text += page_obj.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    metadatas = [{"source": f"{i}-pl"} for i in range(len(text_chunks))]
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.markdown(("User: "+message.content))
        else:
            st.markdown(("AI: "+message.content))


def main():
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if st.session_state.conversation is not None:
        st.header("Ask questions from your PDF:books:")
        user_question = st.chat_input("Ask a question about your documents:")
        if user_question:
            handle_userinput(user_question)

    if st.session_state.conversation is None:
        st.header("Upload your PDF here")
        pdf_doc = st.file_uploader("Browse your file here",type="pdf")
        if pdf_doc is not None:
            with st.spinner("Processing"):
                # get pdf text
                raw_text = extract_text_from_pdf(pdf_doc)
    
                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
    
                # create vector store
                vectorstore = get_vectorstore(text_chunks)
    
                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)

                st.rerun()

if __name__ == '__main__':
    main()