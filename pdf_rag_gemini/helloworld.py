from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import config

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=config.GOOGLE_API_KEY  
)

# Create a simple prompt
prompt = PromptTemplate.from_template("Tell me a funny joke about {topic}")

# Create and run the chain
chain = LLMChain(llm=llm, prompt=prompt)
response = chain.run(topic="programming")

print(response)

