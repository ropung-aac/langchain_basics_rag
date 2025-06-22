import streamlit as st

st.set_page_config(page_title="Hello World", page_icon=":tada:", layout="wide")

st.title("☺️ Hello World")
st.caption("This is a caption")


if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if user_question := st.chat_input(placeholder="Ask me anything..."):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    with st.chat_message("ai"):
        st.write("여기는 AI 답변 입니다.")
    st.session_state.messages.append({"role": "ai", "content": "여기는 AI 답변 입니다."})

print(f"after === {st.session_state.messages}")