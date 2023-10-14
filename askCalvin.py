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

# # # Quick data visualization to ensure chunking was successful

# # # Create a list of token counts
# # token_counts = [count_tokens(chunk.page_content) for chunk in chunks]

# # # Create a DataFrame from the token counts
# # df = pd.DataFrame({'Token Count': token_counts})

# # # Create a histogram of the token count distribution
# # df.hist(bins=40, )

# # # Show the plot
# # plt.show()

# embeddings = OpenAIEmbeddings()

# # Create vector database
# db = FAISS.from_documents(chunks, embeddings)

# # query = "who is Paul?"
# # docs = db.similarity_search(query)
# # # docs[0]

# # chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")

# # query = "Who created transformers?"
# # docs = db.similarity_search(query)

# # chain.run(input_documents=docs, question=query)

# # Create conversation chain that uses our vectordb as retriver, this also allows for chat history management (also does similarity search)
# qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.1), db.as_retriever())

# chat_history = []

# print("Welcome to the Transformers chatbot! Type 'exit' to stop.")

# while True:
#     query = input('Please enter your question: ')
#     if query.lower() == 'exit':
#         print("Thank you for chatting with calvin!")
#         break
#     result = qa({"question": query, "chat_history": chat_history})
#     chat_history.append((query, result['answer']))
#     print(f'Chatbot: {result["answer"]}')