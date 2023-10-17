"""A conversational retrieval chain."""

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from fastapi import FastAPI
from langserve import add_routes
import os

vectorstore = FAISS.from_texts(
    ["cats like fish", "dogs like sticks"],
    embedding=OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
)
retriever = vectorstore.as_retriever()

model = ChatOpenAI()

chain = ConversationalRetrievalChain.from_llm(model,retriever)

"""A server for the chain above."""


app = FastAPI(title="Retrieval App")

add_routes(app, chain)

if _name_ == "_main_":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)