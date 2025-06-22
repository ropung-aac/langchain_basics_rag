from dotenv import load_dotenv
import streamlit as st

load_dotenv()

# from openai import OpenAI

# client = OpenAI()

# response = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[
#         {"role": "user", "content": "대한민국 수도는 어디야?"}
#     ]
# )

# print(response.choices[0].message.content)

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

# res = llm.invoke("대한민국 수도는 어디야?")
# print(res.content)

 
st.write("""
# 내분비내과 의사
## 의사 솔루션
""")

poem = st.text_input("고민을 입력해주세요.")

if st.button("고민 작성하기"):
    with st.spinner("Wait for it...", show_time=True):
        res = llm.invoke(poem + " 다이어트 솔루션을 컨설팅해주세요.")
        st.write(res.content)