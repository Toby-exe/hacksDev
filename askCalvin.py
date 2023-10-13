import os 
import sys
from langchain.chains import LLMChain
import constants
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
from flask import Flask, request, jsonify

cache = InMemoryCache()
set_llm_cache(cache)

os.environ["OPENAI_API_KEY"] = constants.APIKEY

query = sys.argv[1]
print("Query: ", query)


loader = TextLoader('data.txt')
index = VectorstoreIndexCreator().from_loaders([loader])

print(index.query(query))