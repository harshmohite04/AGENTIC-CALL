from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.schema import HumanMessage

# Your Hugging Face API token
import os
from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize the endpoint
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    huggingfacehub_api_token=HF_TOKEN,
    temperature=0,
    max_new_tokens=512
)

# Wrap it as a chat model (optional)
chat_model = ChatHuggingFace(llm=llm)

# Example chat
response = chat_model.invoke([HumanMessage(content="Write a short poem about AI and nature.")])

print(response.content)
