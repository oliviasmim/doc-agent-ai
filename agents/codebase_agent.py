from crewai import Crew, Task, Agent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os

load_dotenv()
openai_llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), model="gpt-3.5-turbo")

codebase_agent = Agent(
    role="Codebase Analyst",
    goal="Extract clear, structured info from codebase for documentation purposes",
    backstory="An expert in analyzing code repositories and extracting API structures and important comments.",
    llm=openai_llm,
)

docs_writer_agent = Agent(
    role="Technical Writer",
    goal="Generate clear markdown documentation",
    backstory="You write precise, helpful documentation from structured codebase information.",
    llm=openai_llm,
)

crew = Crew(
    agents=[codebase_agent, docs_writer_agent],
    tasks=[
        Task(
            description="Extract a structured outline of APIs, functions, and classes from the repo.",
            expected_output="A detailed JSON structure containing all APIs, functions, and classes with their descriptions, parameters, and return values.",
            agent=codebase_agent,
        ),
        Task(
            description="Create markdown documentation from structured outlines.",
            expected_output="Complete markdown documentation covering all components of the codebase in a clear, organized format.",
            agent=docs_writer_agent,
            output_file='./docs/README.md'
        ),
    ],
    verbose=True
)

result = crew.kickoff()
print(result)