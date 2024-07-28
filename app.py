import streamlit as st
import os
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
import numpy as np
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

st.title("Job Application Matcher with GPT-4 and FAISS")

# Using the most capable OpenAI LLM (gpt-4)
llm = ChatOpenAI(model="gpt-4")

def vector_embedding():
    if "vector" not in st.session_state:
        with st.spinner("Loading and embedding documents..."):
            st.session_state.embeddings = OpenAIEmbeddings()
            st.session_state.loader = PyPDFDirectoryLoader("./data")
            st.session_state.docs = st.session_state.loader.load()
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
            st.session_state.vector_ready = True
            st.success("Vector Store DB is Ready")

# Adjusted prompt to filter resumes with at least 75% match
prompt = ChatPromptTemplate.from_template(
    """
    You are a job matching assistant. Based on the job description provided, identify and return resumes that match at least 75% of the job requirements.
    <context>
    {context}
    </context>
    Job Description: {input}
    List the resumes that match at least 75% of the job description.
    """
)

st.button("Load Data and Embed Documents", on_click=vector_embedding)

if 'vector_ready' in st.session_state and st.session_state.vector_ready:
    prompt1 = st.text_area("Enter the Job Description")

    if st.button("Match Applications"):
        import time

        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        if prompt1:
            start = time.process_time()
            response = retrieval_chain.invoke({'input': prompt1})
            st.write("Response time:", time.process_time() - start)

            # Compute similarity between job description and documents
            job_description_embedding = st.session_state.embeddings.embed_query(prompt1)
            similarities = []

            for doc in st.session_state.final_documents:
                doc_embedding = st.session_state.embeddings.embed_documents([doc.page_content])[0]
                similarity = np.dot(job_description_embedding, doc_embedding) / (np.linalg.norm(job_description_embedding) * np.linalg.norm(doc_embedding))
                similarities.append((similarity, doc))

            # Filter documents based on a similarity threshold
            matched_docs = [(sim, doc) for sim, doc in similarities if sim >= 0.75]

            if matched_docs:
                st.write("Matched Applications:")
                for i, (sim, doc) in enumerate(matched_docs):
                    if st.button(f"Show Application {i+1} (Match: {sim*100:.2f}%)"):
                        st.write(f"Application {i+1}:")
                        st.write(doc.page_content)
                        st.write("----------------------------------------------")
            else:
                st.write("No applications matched the criteria.")
else:
    st.write("Please initialize the document embeddings first.")
