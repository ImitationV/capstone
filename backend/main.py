from fastapi import FastAPI
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import os

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

app = FastAPI()

# Define the expected format of the output
class FinancialAdvice(BaseModel):
    response: str = Field(..., description="Helpful financial advice based on user's query.")

    @classmethod
    def schema(cls):
        schema = super().schema()
        return schema

class Query(BaseModel):
    message: str = Field(..., max_length=1000)

# Setup LangChain parser and model
parser = PydanticOutputParser(pydantic_object=FinancialAdvice)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
        You are a helpful financial advisor. Provide clear and actionable financial guidance 
        in response to user questions. Keep answers concise, practical, and easy to understand.
        Avoid legal disclaimers or lengthy disclaimers. Try to find sources of your answer. If 
        you cant find any source mention the user that it is from your assumption.
        Format your response as follows:
        {format_instructions}
    """),
    ("human", "{user_message}")
])

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7
)

chain = prompt | llm | parser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat(query: Query):
    try:
        result = chain.invoke({
            "user_message": query.message,
            "format_instructions": parser.get_format_instructions()
        })
        return {"response": result.response}
    except Exception as e:
        return {"error": str(e)}
