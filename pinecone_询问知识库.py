from langchain.vectorstores import Chroma,Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
import os
import streamlit as st
st.title('开山股份本地知识库')
user_input = st.text_input("请输入你想了解的关于开山股份的信息（输入p时结束）：")
OPENAI_API_KEY=st.text_input("请输入你的OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings=OpenAIEmbeddings()
pinecone.init(api_key='a4dfd6f5-79ce-438c-8628-855ef4d8480b',environment='us-west4-gcp-free')
index_name='wtfgpt'

docsearch = Pinecone.from_existing_index(index_name, embeddings)

from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo,
)
from langchain.llms import OpenAI
llm=OpenAI(temperature=1)
vectorstore_info = VectorStoreInfo(
    name="开山股份知识库",
    description="当询问到关于开山股份的问题时非常有用，输入必须是一个问题",
    vectorstore=docsearch,
)
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)
agent_executor = create_vectorstore_agent(llm=llm, toolkit=toolkit, verbose=True)

st.write(agent_executor.run(user_input))
# while True:
#     user_input = input("请输入你想了解的关于开山股份的信息（输入p时结束）：")
#     if user_input == "p":
#         print("AI咨询停止！")
#         break
#     else:
#         print(agent_executor.run(user_input+'请用中文回答'))
