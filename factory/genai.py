# genai.py
import streamlit as st
import os
#import openai_helper
import csv

# Set the title and subtitle of the app
st.title('🦜🔗 PDF-Q/A CHATBOT')
question = st.text_input('Input your question here')

# If the user hits enter
if question:
    qa = openai_helper.qa(question)
    # ...and write it out to the screen
    result = qa
    st.write(result)
    
    # Log user question and model answer to a CSV file
    with open('qa_log.csv', 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([question, result])

# openai_helper.py
import os
import tiktoken
import chromadb

from langchain.document_loaders import OnlinePDFLoader, UnstructuredPDFLoader, PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import ChatOpenAI

os.environ['OPENAI_API_KEY'] = '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'

# Text Splitter to Chunks
loader = PyPDFLoader("/q3-2022-corporate-overview.pdf")
pdfData = loader.load()
text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
splitData = text_splitter.split_documents(pdfData)

# Embedding and storing in VectorDB
openai_key=os.environ.get('OPENAI_API_KEY')
embeddings = OpenAIEmbeddings()
vectDB = Chroma.from_documents(splitData,embeddings)

def qa(question):
    # Creating Prompt Template
    custom_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.If you do not know the answer, reply with "I am sorry, I don't Know"

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

    # Defining Buffer Memory and extracting context from vector store
    llm = ChatOpenAI(temperature=0.2)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), vectDB.as_retriever(), condense_question_prompt=CUSTOM_QUESTION_PROMPT, memory=memory)
    result = qa_chain({"question": question})
    result = result["answer"]
    return result
