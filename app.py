from flask import Flask, render_template, request, jsonify


# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch


# tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
# model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")


app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_Chat_response(input)


def get_Chat_response(query):

    # import os 
    # import sys
    # from langchain.chains import LLMChain
    # import constants
    # from langchain.document_loaders import TextLoader
    # from langchain.indexes import VectorstoreIndexCreator
    # from langchain.cache import InMemoryCache
    # from langchain.globals import set_llm_cache
    # from flask import Flask, request, jsonify
    # # from flask import Flask, request, jsonify

    
    # cache = InMemoryCache()
    # set_llm_cache(cache)

    # os.environ["OPENAI_API_KEY"] = constants.APIKEY

    # # query = sys.argv[1]
    # print("Query: ", query)


    # loader = TextLoader('data.txt')
    # index = VectorstoreIndexCreator().from_loaders([loader])

    # return index.query(query)

    # import os
    # import constants
    # import sys
    # import pandas as pd
    # import matplotlib.pyplot as plt
    # from transformers import GPT2TokenizerFast
    # from langchain.document_loaders import TextLoader
    # from langchain.text_splitter import RecursiveCharacterTextSplitter
    # from langchain.embeddings import OpenAIEmbeddings
    # from langchain.vectorstores import FAISS
    # from langchain.chains.question_answering import load_qa_chain
    # from langchain.llms import OpenAI
    # from langchain.chains import ConversationalRetrievalChain
    # import textract
    # from langchain.globals import set_llm_cache
    # from langchain.cache import InMemoryCache


    # # from flask import Flask, request, jsonify
    # cache = InMemoryCache()
    # set_llm_cache(cache)

    # os.environ["OPENAI_API_KEY"] = constants.APIKEY

    # loader = TextLoader('data.txt')

    # with open('data.txt', 'r') as file:
    #     text = file.read()

    # tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # def count_tokens(text: str) -> int:
    #     return len(tokenizer.encode(text))

    # text_splitter = RecursiveCharacterTextSplitter(
    #     # Set a really small chunk size, just to show.
    #     chunk_size = 512,
    #     chunk_overlap  = 24,
    #     length_function = count_tokens,
    # )

    # chunks = text_splitter.create_documents([text])
    # type(chunks[0]) 
    # embeddings = OpenAIEmbeddings()

    # # Create vector database
    # db = FAISS.from_documents(chunks, embeddings)

    # qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.1), db.as_retriever())

    # chat_history = []

    # result = qa({"question": query, "chat_history": chat_history})
    # return result["answer"]
    return query

if __name__ == '__main__':
    app.run()
