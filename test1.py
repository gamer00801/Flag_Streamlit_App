import requests
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
import os
from dotenv import load_dotenv
# Load environment variables
load_dotenv()
# 設定存取權杖和檔案ID

access_token = os.environ.get('ACCESS_TOKEN')
file_id = os.environ.get('FILE_KEY')
ids = os.environ.get('NODE_IDS')
# 發送API請求
headers = {"content-type": "application/json", "Accept-Charset": "UTF-8", "X-FIGMA-TOKEN": access_token}
fileURL = 'https://api.figma.com/v1/files/' + file_id + ids
response = requests.get(fileURL, headers=headers)


# 處理API回應
if response.status_code == 200:
    data = response.json()
    with open("file_name.txt", "w") as file:
        file.write(data)
    # 在這裡處理返回的資料
else:
    print("請求失敗:", response.status_code)
loader = TextLoader(file, encoding='utf-8')