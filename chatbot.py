import os
import re
from langchain.document_loaders import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.chains import RetrievalQA
import streamlit as st
from dotenv import load_dotenv
from tempfile import NamedTemporaryFile
import streamlit.components.v1 as components
from streamlit_chat import message
from langchain import LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
import pandas as pd
# Load environment variables from a .env file (containing OPENAI_API_KEY)
load_dotenv()

# Set the OpenAI API key and dataset path from the environment variables
os.environ["OPENAI_API_KEY"] == st.secrets["OPENAI_API_KEY"]

st.title('FlagBot')

# Define the chatbot template
template = """請根據上下文來回答問題,盡量針對資料內的重點做回覆,如果你不知道答案，就說你不知道，不要試圖編造答案。
{context}
You是專業客服人員,使用繁體中文,對問題會盡力回答,回答問題後會在詢問是否還有什麼問題:
You:你好我是客服人員
me:{question}
"""

# Define the chatbot prompt
prompt = PromptTemplate(
    input_variables=["context", "question"], 
    template=template
)

# Initialize the LLMChain for the chatbot
chatgpt_chain = LLMChain(
    llm=ChatOpenAI(temperature=0, request_timeout=120), 
    prompt=prompt, 
    verbose=True, 
    memory=ConversationBufferWindowMemory(k=2),
)

# Function to read the HTML content from the specified file
def read_html_content(file_name):
    with open(file_name, "r",encoding='UTF-8') as file:
        html_content = file.read()
    return html_content

# Function to write the HTML content to the specified file
def write_html_content(html_content):
    with open(updated_file_name, "w") as file:
        file.write(html_content)

# Function to check if the output is a valid HTML/CSS code update
def is_valid_html_css_code(output):
    pattern = re.compile(r'<.*?>|{.*?}')
    return bool(pattern.search(output))

# Function to generate a response using OpenAI's ChatCompletion API
def generate_response(prompt, html_content):
    completion = chatgpt_chain.predict(
        human_input=(
            "You are a senior developer. I want you to improve the following code with the following {prompt}: {html_content}. Only output new html/css code. Do not write additional text."
            "The code is html/css that was created by AI based on a Figma document."
        ).format(prompt=prompt, html_content=html_content)
    )
    return completion

# Function to render the HTML content using Streamlit's markdown function
def render_html(html_content):
    components.html(html_content, scrolling=True, height=300)

# Function to get the user's input from the text input field
def get_text():
    # Create a Streamlit input field and return the user's input
    input_text = st.text_input("", key="input")
    return input_text

# Initialize the session state for generated responses and past inputs
if 'generated' not in st.session_state:
    st.session_state['generated'] = ['您好,我是客服小精靈,有什麼需要協助的嗎?']

if 'past' not in st.session_state:
    st.session_state['past'] = ['Hello']

prompt="""請根據上下文來回答問題,盡量針對資料內的重點做回覆,如果你不知道答案，就說你不知道，不要試圖編造答案。
{context}
You是專業客服人員,使用繁體中文,對問題會盡力回答,回答問題後會在詢問是否還有什麼問題:
:你好我是客服人員
me:{question}
"""
PROMPT = PromptTemplate(
    template=prompt, input_variables=["context","question"]
)
a=st.file_uploader(label="上傳")
def streamlit_file(a):
    df = pd.read_csv(a,engine='python',encoding='utf-8')
    st.dataframe(df)
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(a.getvalue())
        tmp_file_path = tmp_file.name
    loader = CSVLoader(file_path=tmp_file_path,encoding='utf-8')
    doc=loader.load_and_split()
    embeddings = OpenAIEmbeddings()
    dataset_path = '/local_path'
    db = DeepLake.from_documents(doc, embeddings, dataset_path=dataset_path)
    retriever = db.as_retriever()
    retriever.search_kwargs['distance_metric'] = 'cos'
    retriever.search_kwargs['k'] = 7
    model = ChatOpenAI(model_name='gpt-3.5-turbo') # 'gpt-3.5-turbo',
    chain_type_kwargs = {"prompt":PROMPT}
    qa = RetrievalQA.from_chain_type(model, chain_type="stuff", retriever=retriever, chain_type_kwargs=chain_type_kwargs)
    return qa
# Render the initial HTML content
# display = render_html(read_html_content(file_name))
if a is not None:
    qa=streamlit_file(a)
# Get the user's input from the text input field
user_input = get_text()

# If there is user input, search for a response and update the HTML content
if user_input:
    output=qa.run(user_input)
    # Append the user input and generated output to the session state
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

# If there are generated responses, display the conversation using Streamlit messages
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])):
        message(st.session_state['past'][i],
                is_user=True, key=str(i) + '_user')
        message(st.session_state["generated"][i], key=str(i))


