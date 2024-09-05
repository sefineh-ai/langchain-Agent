import os
import re
import time
import pandas as pd
from dotenv import load_dotenv
from typing import List, Optional
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_astradb import AstraDBVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter


load_dotenv()
class ScholarshipDataProcessor:
    """Handles the processing of both structured and unstructured scholarship data."""

    def __init__(self, structured_file_path: str, unstructured_file_path: str) -> None:
        self.structured_file_path = structured_file_path
        self.unstructured_file_path = unstructured_file_path
        self.structured_data: Optional[pd.DataFrame] = None
        self.unstructured_data: Optional[str] = None
        self.cleaned_unstructured_data: Optional[str] = None
        self.cleaned_structured_data: Optional[List[Document]] = None

    @staticmethod
    def clean_data(data: str) -> str:
        """Cleans input data by removing non-ASCII characters, non-printable characters, and excessive whitespace."""
        data = data.encode("ascii", "ignore").decode("ascii")
        data = "".join(char for char in data if char.isprintable())
        data = re.sub(r"\s+", " ", data)
        data = re.sub(r"\n+", "\n", data)
        data = re.sub(r"\t+", "\t", data)
        return data.strip()

    def load_unstructured_data(self) -> None:
        """Loads and cleans unstructured data from a file."""
        with open(self.unstructured_file_path, "r", encoding="utf-8", errors="ignore") as file:
            unstructured_lines: List[str] = [self.clean_data(line) for line in file.readlines()]
            self.cleaned_unstructured_data = " ".join(unstructured_lines)

    def load_structured_data(self) -> None:
        """Loads structured data from a CSV file and cleans it."""
        self.structured_data = pd.read_csv(self.structured_file_path)
        for column in self.structured_data.columns:
            self.structured_data[column] = self.structured_data[column].apply(self.clean_data)

    def create_documents_from_unstructured_data(self) -> List[Document]:
        """Splits unstructured data into chunks and creates Document objects."""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=0)
        return [Document(page_content=text) for text in text_splitter.split_text(self.cleaned_unstructured_data or "")]

    def create_documents_from_structured_data(self) -> List[Document]:
        """Combines columns of structured data into a single string for each record and creates Document objects."""
        documents: List[Document] = []
        for _, row in self.structured_data.iterrows():
            document_content = " ".join(row.values.astype(str))
            documents.append(Document(page_content=document_content))
        return documents

    def process(self) -> None:
        """Main processing function to load, clean, and prepare data."""
        self.load_unstructured_data()
        self.load_structured_data()
        self.cleaned_structured_data = self.create_documents_from_structured_data()


class VectorStoreManager:
    """Manages interactions with the AstraDB Vector Store."""

    def __init__(self) -> None:
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = self._initialize_vectorstore()

    def _initialize_vectorstore(self) -> AstraDBVectorStore:
        """Initializes the AstraDB Vector Store with configuration from environment variables."""
        return AstraDBVectorStore(
            embedding=self.embeddings,
            collection_name="sfsu",
            api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
            token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
            namespace=os.getenv("ASTRA_DB_NAMESPACE", None),
        )

    def add_documents(self, documents: List[Document]) -> None:
        """Adds a list of documents to the vector store."""
        self.vectorstore.add_documents(documents)

    def similarity_search(self, query: str) -> List[Document]:
        """Performs a similarity search in the vector store."""
        return self.vectorstore.similarity_search(query)


def main() -> None:
    """Main function to execute the script."""
    start_time: float = time.time()

    # File paths
    structured_file_path = "data/structured_scholarship_data.csv"
    unstructured_file_path = "data/unstructured_scholarship_data.txt"

    # Process scholarship data
    processor = ScholarshipDataProcessor(structured_file_path, unstructured_file_path)
    processor.process()

    # Manage vector store
    vector_manager = VectorStoreManager()
    vector_manager.add_documents(processor.create_documents_from_unstructured_data())
    response = vector_manager.similarity_search("What is the scholarship process for campus partners?")
    print(response)

    end_time: float = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
