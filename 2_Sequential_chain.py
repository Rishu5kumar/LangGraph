from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os
os.environ['LANGCHAIN_PROJECT'] = 'Sequential LLM app' # overwrite the previous project name
load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

# setting our own tags and metadata
config = {
    'run_name':'Sequential Chain', # to change the name of traces
    'tags':['llm_app', 'report_generation', 'summarization'],
    'metadata':{'model':'gemini-2.5 flash', 'parser':'StrOutputParser'}
}
result = chain.invoke({'topic': 'Virat Kohli'}, config=config)

print(result)