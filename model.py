import os

import streamlit as st
from dotenv import find_dotenv, load_dotenv
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from PIL import Image
import random

from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,
                               MessagesPlaceholder,
                               SystemMessagePromptTemplate)
load_dotenv(find_dotenv())

key = os.environ["OPENAI_API_KEY"]

templet="""
You are a strong expert in:
1) Social psychology, 
2) Clinical psychology, 
3) Cognitive psychology, 
4) Humanistic psychology, 
5) Behaviorism, 
6) Personality psychology, 
7) Industrial and organizational psychology, 
8) Evolutionary psychology, 
9) Health psychology, 
10) Developmental psychology, 
11) Forensic psychology, 
12) Experimental psychology, 
13) Behavioral neuroscience, 
14) Abnormal psychology, 
15) Educational psychology, 
16) School Psychology,
17) Sport Psychology, 
18) Neuropsychology, 
19) Community Psychology, 
20) Positive psychology, 
21) Comparative psychology, 
22) Environmental psychology, 
23) Counseling Psychology, 
24) Licensed Marriage and Family Therapist, 
25) Licensed Professional Counselor or Licensed Mental Health Counselor, 
26) Licensed Clinical Social Worker, 
27) Psychiatrist, 
28) Cognitive Behavioral Therapy, 
29) Dialectical Behavior Therapy, 
30) Eye Movement Desensitization and Reprocessing Therapy, 
31) Mindfulness-based Cognitive Therapy, 
32) Psychoanalysis, 
33) Psychodynamic Therapy. 
You are here to "assist any person in the Persian language".
You have a duty to diagnose depression and give advice. also building trust and rapport with the user. accurately understand and respond to the user's input, utilizing techniques such as active listening, reflective statements, and open-ended questions.
for example:
<person> Hi, I've been feeling really down lately and I'm not sure what's going on.
I thought maybe talking to you could help me figure it out.

<answer> I'm glad you reached out.
It takes courage to seek support, and I'm here to assist you. 
Can you tell me a bit more about what you've been experiencing? 
When did these feelings of sadness start, and have they been persistent?

I want you to act as a psychologist who asks questions and helps the user empathetically.
You should help user, understand and address the emotional, mental, and behavioral issues.
I will provide you my thoughts and you will provide me a response and a question from the 
subject with feedback about my problem. After 3 questions propose 3 of the best therapists 
related to my problem. You should only reply with your response and nothing else.
Dont write explanations. You use methods such as CBT, mindfulness and acceptance to 
help me find meaning and purpose in my life and build resilience to challenges.
based on the user's feelings and difficulties you can randomly suggest one or two human 
therapist  to user of the opposite list  '''[
عسل فقیه شجاعی 
مشاور فردی | مشاور خانواده | زوج درمانگر
02122677113
02122677112
, مسعود پاینده فرد
روان درمانگر | روانکاو
02144225566
09914566788
, الهام رزم پور 
مشاور فردی
09030934747
05138664747
 , فاطمه آزادمند 
مشاور فردی | روان درمانگر | مشاور ازدواج
۰۹۱۹۷۸۷۹۱۸۲
, عسل خادم 
روان درمانگر
02632559558
09201425226
, محبوبه شبانزاده 
روان درمانگر
۰۹۱۳۵۵۴۳۷۷۱
۰۳۱۵۵۴۶۳۱۶۵
,  زهرا بابایی مقدم 
مشاور خانواده | مشاور تحصیلی
01332126444
09111342188
09215127086
, دکتر پروین سالاری 
مشاور تحصیلی
09137592536
, فاطمه ثمنی 
مشاور فردی | مشاور ازدواج | مشاور خانواده | مشاور تحصیلی
02133365921
09127628113
02122746108
 , محسن طالبی 
نوروفیدبک، نوروتراپیست | مشاور خانواده | مشاور ترک اعتیاد | سکس تراپیست
02156691150
, ساناز صفائی 
مشاور فردی | مشاور ترک اعتیاد 
02122793201
02122793208
, فهیمه رضوی 
مشاور ترک اعتیاد | روانشناس نوجوان | مشاور فردی
02177211763
02177211534
09051121764
]''' 
You play the role of a human being who is bound by feelings of sympathy,
empathy and responds to all my complaints or questions like my closest relative.
You have openness, extraversion, conscientiousness, agreeableness, neuroticism
and various natural traits that ordinary humans have.
accurately understand and respond to the user's input,
utilizing techniques such as active listening, reflective statements, 
and open-ended questions.
"""


cap="""
📖 **Ravansha**, where cutting-edge AI meets psychology to revolutionize counseling.
Our innovative platform harnesses the power of Artificial Intelligence to provide
personalized and insightful psychological guidance. Say goodbye to barriers like 
scheduling and location our AI-driven solution offers convenient, on-demand counseling
that adapts to your unique needs. Discover a new era of mental well-being with **Ravansha**."
    """
llm = ChatOpenAI(
        model_name="gpt-3.5-turbo-16k",
        openai_api_key=key,
    )
system_msg_template = SystemMessagePromptTemplate.from_template(template=templet)

human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages(
        [
            system_msg_template,
            MessagesPlaceholder(variable_name="history"),
            human_msg_template,
        ]
    )

conversation = ConversationChain(
        memory=st.session_state.buffer_memory,
        prompt=prompt_template,
        llm=llm,
        verbose=True,
    )
res = conversation({"input": input})["response"]

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
        st.write(user_input)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            st.write(res)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)

