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
    return query

if __name__ == '__main__':
    app.run()
