### 🧠 Building AI Research Assistant Agent

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import Tool
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from datetime import datetime

# ============================
# 🌿 1. Load environment
# ============================
load_dotenv()

# ============================
# 🧱 2. Define Output Schema
# ============================
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# ============================
# 🧠 3. LLM Configuration
# ============================
llm = ChatOpenAI(model="gpt-4o-mini")

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant.
            Always provide the final answer strictly in valid JSON format as per the schema.
            Use tools when needed but make sure the last message is only the structured JSON output.
            Do not add any extra text, explanation, or markdown.
            {format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# ============================
# 🔧 4. Define Tools
# ============================
# Search tool
search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="Search",
    func=search.run,
    description="Search the web for recent information."
)

# Wikipedia tool
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

# Custom save tool (used after agent completes)
def save_to_txt(data: str, filename: str = "research_output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    return f"✅ Data successfully saved to {filename}"

save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Saves structured research data to a text file."
)

# ============================
# 🤖 5. Agent Creation
# ============================
tools = [search_tool, wiki_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)

# ============================
# 🧪 6. Run Interaction
# ============================
query = input("What can I help you research? ")

raw_response = executor.invoke({"query": query})

# ============================
# 🧭 7. Parse Output Safely
# ============================
try:
    structured_response = parser.parse(raw_response.get("output", ""))
    print("\n✅ Parsed Structured Response:\n", structured_response)
except Exception as e:
    print("\n❌ Error parsing response:", e)
    print("📝 Raw Output:\n", raw_response.get("output", ""))

# ============================
# 💾 8. Optional Save to File
# ============================
save_choice = input("\nDo you want to save the response to a file? (y/n): ").strip().lower()
if save_choice == "y":
    save_result = save_to_txt(raw_response.get("output", ""))
    print(save_result)
else:
    print("❎ Skipping file save.")
