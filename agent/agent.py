import os
import re
import pandas as pd
from dotenv import load_dotenv
from typing import Optional, List
from langchain_core.tools import Tool
from pydantic import BaseModel, Field
from langchain.agents import AgentExecutor
from langchain_core.documents import Document
from langchain_astradb import AstraDBVectorStore
from langchain_core.prompts import MessagesPlaceholder
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_community import GoogleSearchAPIWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)

load_dotenv()

ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_NAMESPACE = os.getenv("ASTRA_DB_NAMESPACE")
GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:90.0) Gecko/20100101 Firefox/90.0"
}


class SearchQuery(BaseModel):
    query: str = Field(description="Search query.")


class SFResourceMatching:
    def __init__(self) -> None:
        """Initialize the ScholarshipAgent with environment configurations and services."""
        self.ASTRA_DB_APPLICATION_TOKEN: Optional[str] = os.getenv(
            "ASTRA_DB_APPLICATION_TOKEN"
        )
        self.ASTRA_DB_API_ENDPOINT: Optional[str] = os.getenv("ASTRA_DB_API_ENDPOINT")
        self.ASTRA_DB_NAMESPACE: Optional[str] = os.getenv("ASTRA_DB_NAMESPACE") or None
        self.SERPER_API_KEY: Optional[str] = os.getenv("SERPER_API_KEY")
        if not all([self.ASTRA_DB_APPLICATION_TOKEN, self.ASTRA_DB_API_ENDPOINT]):
            raise ValueError(
                "Environment variables for Astra DB are not set correctly."
            )

        self.embeddings: OpenAIEmbeddings = OpenAIEmbeddings()
        self.vectorstore: AstraDBVectorStore = AstraDBVectorStore(
            embedding=self.embeddings,
            collection_name="sfsu",
            api_endpoint=self.ASTRA_DB_API_ENDPOINT,
            token=self.ASTRA_DB_APPLICATION_TOKEN,
            namespace=self.ASTRA_DB_NAMESPACE,
        )
        self.model: ChatOpenAI = ChatOpenAI(model_name="gpt-4", temperature=0)
        MEMORY_KEY = "chat_history"
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are very powerful assistant, but bad at calculating lengths of words.",
                ),
                MessagesPlaceholder(variable_name=MEMORY_KEY),
                ("user", "{query}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

    def clean_data(self, data: str) -> str:
        """
        Clean and preprocess text data to ensure it is printable and well-formatted.
        Args:
            data (str): The raw text data to clean.
        Returns:
            str: The cleaned and formatted text data.
        """
        if not isinstance(data, str):
            raise TypeError("Input data must be a string.")

        data = data.encode("ascii", "ignore").decode("ascii")
        data = "".join(char for char in data if char.isprintable())
        data = re.sub(r"\s+", " ", data)
        data = re.sub(r"\n+", "\n", data)
        data = re.sub(r"\t+", "\t", data)
        data = data.strip()
        return data

    def process_and_store_documents(self, text_data: str) -> None:
        """
        Process text data into documents and store them in the vector store.
        Args:
            text_data (str): The raw text data to process and store.
        """
        if not isinstance(text_data, str):
            raise TypeError("Text data must be a string.")

        cleaned_data: str = self.clean_data(text_data)

        text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=0
        )
        split_texts: List[str] = text_splitter.split_text(cleaned_data)

        documents: List[Document] = [
            Document(page_content=text) for text in split_texts
        ]
        self.vectorstore.add_documents(documents)
        print("Documents have been stored in the vector store successfully.")

    def query_vectorstore(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Query the vector store with a user query to find similar documents.
        Args:
            query (str): The user query for searching the vector store.
            top_k (int): The number of top results to retrieve. Default is 5.
        Returns:
            List[Document]: List of documents that are most similar to the query.
        """
        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        documents: List[Document] = self.vectorstore.similarity_search(
            query, top_k=top_k
        )
        return documents

    def process_and_store_structured_data(
        self, csv_file_path: Optional[str] = "structured_scholarship_data.csv"
    ) -> None:
        """
        Process text data into structured documents and store them in the vector store.
        Args:
            text_data (str): The raw text data to process and store.
        """
        structured_data_df: pd.DataFrame = pd.read_csv(csv_file_path)
        structured_data_df = structured_data_df.astype(str)
        for colum in structured_data_df.columns:
            structured_data_df[colum] = structured_data_df[colum].apply(self.clean_data)

        structured_data_documents: List[Document] = []
        for index, row in structured_data_df.iterrows():
            document: Document = Document(page_content=" ".join(row.values.astype(str)))
            structured_data_documents.append(document)
        self.vectorstore.add_documents(structured_data_documents)
        print("Structured documents have been stored in the vector store successfully.")

    def read_text_data(self, file_path: str) -> str:
        """
        Read text data from a file and return it as a string.
        Args:
            file_path (str): The path to the text file.
        Returns:
            str: The contents of the text file as a string.
        """
        if not isinstance(file_path, str):
            raise TypeError("File path must be a string.")

        with open(file_path, "r") as file:
            text_data: str = file.read()
        return text_data

    def create_tools(self) -> List[Tool]:
        """
        Create a tool for searching the web.
        Args:
          None
        Return:
          Tool: Tool for searching the web
        """
        search = GoogleSearchAPIWrapper()

        def top5_results(query):
            return search.results(query, 5)

        search_tool = Tool(
            name="GoogleSearchSnippets",
            description="Search Google for recent results.",
            func=top5_results,
            args_schema=SearchQuery,
        )
        tools = [search_tool]
        return tools

    def create_agent(self, llm_with_tools: any):
        """
        Create an agent for searching the web.
        Args:
          tools: List of tools for the agent
        Return:
          Agent: Agent for searching the web
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Use the following context to answer the user's question: {context}",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "user",
                    "Could you help me to answer my question? Here is my question:\n {query}",
                ),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        agent = (
            {
                "query": lambda x: x["query"],
                "context": lambda x: "\n".join(
                    [
                        document.page_content
                        for document in self.get_context_from_vstore(x["query"])
                    ]
                ),
                "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                    x["intermediate_steps"]
                ),
                "chat_history": lambda x: x["chat_history"],
            }
            | prompt
            | llm_with_tools
            | OpenAIToolsAgentOutputParser()
        )
        return agent

    def create_agent_executor(self, agent: any, tools: List[Tool]):
        """
        Create an agent for searching the web.
        Args:
          None
        Return:
          Agent: Agent for searching the web
        """
        agent_executor = AgentExecutor(
            agent=agent, tools=tools, handle_parsing_errors=True, verbose=True
        )

        return agent_executor

    def read_csv(self, csv_file_path: str) -> str:
        """
        Read a CSV file and return its contents as a string.
        Args:
            csv_file_path (str): The path to the CSV file.
        Returns:
            str: The contents of the CSV file as a string.
        """
        if not isinstance(csv_file_path, str):
            raise TypeError("CSV file path must be a string.")

    def get_context_from_vstore(self, query: str):
        """
        Retrieve documents from the vector store based on a user query.
        Args:
            query (str): The user query for searching the vector store.
        Returns:
            List[Document]: List of documents that are most similar to the query.
        """
        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        documents: List[Document] = self.vectorstore.similarity_search(query, top_k=5)
        return documents


