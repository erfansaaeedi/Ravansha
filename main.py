import os

import streamlit as st
from dotenv import find_dotenv, load_dotenv
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from backend import generate_response
from prompt import prompt , intro
from PIL import Image
import random

load_dotenv(find_dotenv())

key = os.environ["OPENAI_API_KEY"]

st.set_page_config(page_title="🤖 🧠 💬 Ravansha")

# login to profile
with st.sidebar:
    st.title("🙍‍♂️ 🧠 🤖 Ravansha")
    img = Image.open('ravan.jpg')
    st.write('ابتدا فرم را تکمیل کنید تا با روانشا صحبت کنبد ')
    st.image(img)
    add1=random.randint(0,100)
    add2=random.randint(0,100)
    if "captcha" == (add1+add1):
        st.success("Login credentials already provided!", icon="✅")
        hf_captcha = st.secrets["captcha"]
    else:
        hf_name = st.text_input("Enter name : ")
        hf_family = st.text_input("Enter family: ")
        hf_pass = st.text_input("Enter the result: ")
        st.write(f'{add1} + {add2} = ?')
        st.slider('Rate Me ⭐️',0,5,1)
        if not (hf_pass):
            st.warning("Please enter your credentials!", icon="⚠️")
        else:
            st.success("Proceed to entering your prompt message!", icon="👉")
    st.markdown(intro())

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "سلام به روانشا خوش آمدید☺️"}
    ]
if "buffer_memory" not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(
        k=7, return_messages=True
    )
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if user_input := st.chat_input(disabled=not (hf_pass)):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input,)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(user_input, prompt(),key)
            st.write(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
