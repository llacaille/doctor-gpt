import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

os.environ["OPENAI_API_KEY"] = st.secrets["OPEN_API_KEY"]

# Create instance of OpenAI LLM
llm = OpenAI(temperature=0.1, verbose=True)
embeddings = OpenAIEmbeddings()

# Load document
loader = TextLoader("chapter66.txt")
documents = loader.load()


def process_document_and_create_retriever(chunk_size=1000, k=2, chunk_overlap=100):
    ## Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_documents(documents)
    ## Select which embeddings we want to use
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    ## Create the vectorestore to use as the index
    db = Chroma.from_documents(texts, embeddings)
    ## Expose this index in a retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    return qa


qa = process_document_and_create_retriever()

st.title("Doctor GPT 🥼")

with st.sidebar:
    st.header("Set up the parameters 🛠️")
    chunk_size = st.slider(
        "Chunk size (controls the max size, in terms of number of characters, of one chunk to be processed by the model)",
        1000,
        10000,
        2000,
        1000,
    )
    chunk_overlap = st.slider(
        "Chunk overlap (specifies how much overlap there should be between chunks)",
        0,
        1000,
        100,
        50,
    )
    k_argument = st.slider(
        "k parameter (top k chunks to be retrieved so as to feed the retriever)",
        1,
        10,
        2,
        1,
    )
    if st.button("Confirm set-up"):
        qa = process_document_and_create_retriever(
            chunk_size, k_argument, chunk_overlap
        )

st.header("Play with the model ⏯️")
prompt = st.text_input("Ask your question about Chapter 66:")
if prompt:
    result = qa({"query": prompt})
    st.write(result["result"])
