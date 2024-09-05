import re
import pandas as pd
from langchain_community.document_loaders import WebBaseLoader
from typing import List


class ScholarshipDataLoader:
    def __init__(self, csv_file_path: str, output_file_path: str, max_line_length: int = 120) -> None:
        """
        Initialize the data loader with required file paths and configurations.

        Args:
            csv_file_path (str): Path to the CSV file containing URLs.
            output_file_path (str): Path to the output file where data will be saved.
            max_line_length (int): Maximum length of lines in the output file.
        """
        self.urls: List[str] = pd.read_csv(csv_file_path)["scholarship_page"].tolist()
        self.output_file_path: str = output_file_path
        self.max_line_length: int = max_line_length

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and format the text by removing extra whitespace.

        Args:
            text (str): Raw text data.

        Returns:
            str: Cleaned text data.
        """
        text = str(text)
        text = text.encode("ascii", "ignore").decode("ascii")
        text = "".join(char for char in text if char.isprintable())
        text = re.sub(r"\n+", "\n", text)
        text = re.sub(r"\t+", "\t", text)
        cleaned_text: str = re.sub(r"\s+", " ", text)

        return cleaned_text.strip()

    def split_long_lines(self) -> None:
        """
        Split long lines in the output file to maintain a maximum line length.
        """
        with open(self.output_file_path, "r") as file:
            lines: List[str] = file.readlines()

        with open(self.output_file_path, "w") as file:
            for line in lines:
                while len(line) > self.max_line_length:
                    split_pos: int = line.rfind(" ", 0, self.max_line_length)
                    if split_pos == -1:
                        split_pos = self.max_line_length
                    file.write(line[:split_pos] + "\n")
                    line = line[split_pos:].lstrip()
                file.write(line)

    def load_data_from_urls(self) -> None:
        """
        Load data from web pages, clean it, and write to the output file.
        """
        for index, url in enumerate(self.urls):
            try:
                loader: WebBaseLoader = WebBaseLoader(url)
                loader.requests_kwargs = {"verify": False}
                data = loader.load()

                if data:  # Ensure data is not empty
                    title: str = data[0].metadata.get("title", "")
                    source: str = data[0].metadata.get("source", "")
                    cleaned_data: str = self.clean_text(data[0].page_content)
                    self.write_to_file(index, title, source, cleaned_data)

            except Exception as e:
                print(f"Failed to load data from URL {url} with error: {e}")

    def write_to_file(self, index: int, title: str, source: str, cleaned_data: str) -> None:
        """
        Write the cleaned data to the output file.

        Args:
            index (int): Index of the URL being processed.
            title (str): Title of the web page.
            source (str): Source URL of the web page.
            cleaned_data (str): Cleaned content of the web page.
        """
        with open(self.output_file_path, "a") as file:
            file.write(f"Title: {title}\nURL: {source}\n{cleaned_data}\n\n")
        print(f"Data from URL {index + 1} has been written to the file.")

    def run(self) -> None:
        """
        Main method to run the data loading and processing pipeline.
        """
        self.load_data_from_urls()
        self.split_long_lines()
        print("Data loading and processing completed.")


if __name__ == "__main__":
    # Initialize the data loader with required paths and configurations
    loader: ScholarshipDataLoader = ScholarshipDataLoader(
        csv_file_path="data/scholarship_pages.csv",
        output_file_path="data/unstructured_cholarship_data.txt",
        max_line_length=120
    )

    # Run the data loader
    loader.run()
