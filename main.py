### Building Agent

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser  
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool




load_dotenv()

### Creating python class for output schema and inheriting from BaseModel

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]



### Setting LLM

llm = ChatOpenAI(model="gpt-4o-mini")

### Output of LLM should be in the format of ResearchResponse class

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", ## msg to llm
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())


tools = [search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools = tools
)


### Building Agent

agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools = tools
)

AgentExecutor = AgentExecutor(agent=agent, tools=[], verose=True)
query = input("What Can I help you Research?")
raw_response = AgentExecutor.invoke({"query": query})

try: 
    structured_response = parser.parse(raw_response["output"])
    print(structured_response)
except Exception as e:
    print("Error parsing response", e. raw_response)

##print(raw_response)
output_text = raw_response["output"]
####llm = ChatAnthropic(model ="claude-3-5-sonnet-20241022")
#response = llm.predict("Who is better test batsman between kohli and williamson.")
#print(response)

### Building Prompt Template

### Printing only the class output

#structured_response = parser.parse(output_text)  
#print(structured_response)