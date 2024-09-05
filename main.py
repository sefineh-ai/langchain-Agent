from typing import List
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor
from agent.agent import SFResourceMatching
from langchain_core.messages import AIMessage, HumanMessage

def store_webdocuments(store):
    # Add unstructured data documents to vectorstore
    store.process_and_store_structured_data(csv_file_path=csv_file_path)
    # Add structured data documents to vectorstore
    store.process_and_store_documents(str(text_data))

    # Data Preprocessing
    store.process_and_store_documents(str(text_data))
    store.process_and_store_structured_data(csv_file_path=csv_file_path)

csv_file_path: str = "data/structured_scholarship_data.csv"
sf_resource_matching: SFResourceMatching = SFResourceMatching()
text_data: any = sf_resource_matching.read_text_data(
    file_path="data/unstructured_scholarship_data.txt"
)


if __name__ == "__main__":
    chat_history = []
    has_question = input("Do you have any questions? (y/n)")
    if has_question.lower() == "n":
        exit(0)
    
    question = input("What is your question?\n---: ")
    if not question:
        print("Please provide a question")
        exit(1)
    
    tools: List[Tool] = sf_resource_matching.create_tools()

    llm_with_tools: any = sf_resource_matching.model.bind_tools(tools)

    agent: any = sf_resource_matching.create_agent(llm_with_tools=llm_with_tools)

    agent_executor: AgentExecutor = sf_resource_matching.create_agent_executor(
        agent=agent, tools=tools
    )

    result = agent_executor.invoke({"query": question, "chat_history": chat_history})
    chat_history.extend(
        [
            HumanMessage(content=question),
            AIMessage(content=result["output"]),
        ]
    )
