# Import necessary libraries 
import vertexai
from google.oauth2 import service_account
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores.chroma import Chroma
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
from langchain_core.documents import Document
# from langchain_experimental.text_splitter import  SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import wikipediaapi
import numpy as np


class rag_app():
    """Base class for the RAG application

    Attributes
    ----------
    PROJECT_ID : str
        GCP project ID

    REGION : str
        GCP region
    
    CREDS_PATH : str
        Path to the GCP service account credentials

    Methods
    -------

    fetch_wiki_page(title: str) -> str
        Fetches a Wikipedia page based on the title

    collecting_docs(titles: list) -> list
        Collecting data from sources and converting them into documents
    
    init_vertexai() -> None
        Initialize VertexAI embeddings

    format_docs(docs: list) -> str
        Format the documents to be used in the RAG model

    embeddings(embedding_function: object, model: str, credentials: object, project: str, region: str) -> object
        Embed the documents using the specified embedding function
    
    splittext(splitter: object, chunk_size: int, chunk_overlap: float, add_start_index: bool, documents: list) -> list
        Split the text into smaller chunks
    
    run() -> None
        Main function to run the RAG application

    main() -> None
        Main function which recursively asks the user for input and runs the RAG application
    
    """

    PROJECT_ID = "cc-sa-sandbox-20200619"  
    REGION = "asia-southeast1"
    CREDS_PATH = "/Users/vancence.ho/Downloads/cc-sa-sandbox-creds.json"  

    def __init__(self) -> None:
        pass

    def fetch_wiki_page(self, title: str) -> str:
        """Fetches a Wikipedia page based on the title.
        
        Parameters
        ----------
        title : str
            Title of the Wikipedia page

        Returns
        -------
        page.text : str
            Text content of the Wikipedia page

        """

        wiki_wiki = wikipediaapi.Wikipedia(user_agent='rag-application/1.0 (vancence.ho@ollion.com)', language='en')
        
        page = wiki_wiki.page(title)
        
        if page.exists():
            return page.text
        else:
            return None
        
    def collecting_docs(self, titles: list) -> list:
        """Collecting data from sources and converting them into documents
        
        Parameters
        ----------
        titles : str
            Titles of the Wikipedia pages to be fetched

        Returns
        -------
        documents : list
            List of documents containing the page content

        """

        documents = []

        for title in titles: 
            content = self.fetch_wiki_page(title)
            if content:
                documents.append(Document(page_content=content, metadata={"title": title, "source": "Wikipedia"}))
            else:
                print(f"Error: Unable to fetch Wikipedia page for '{title}'")

        return documents

    def init_vertexai(self) -> object:
        """Initialize VertexAI embeddings

        Returns
        -------
        credentials : object
            GCP service account credentials

        Throws
        ------
        Exception
            If unable to load GCP service account credentials

        """

        try: 
            credentials = service_account.Credentials.from_service_account_file(self.CREDS_PATH, scopes=["https://www.googleapis.com/auth/cloud-platform"],)
        except:
            print("Error: Unable to load GCP service account credentials!")

        vertexai.init(credentials=credentials, project=self.PROJECT_ID, location=self.REGION)
        
        print("VertexAI initialized!")
        return credentials

    def format_docs(self, docs: list) -> str:
        """Format the documents to be used in the RAG model
        
        Parameters
        ----------
        docs : list
            List of documents to be formatted

        Returns
        -------
        str

        """
        return "\n\n".join(doc.page_content for doc in docs)
    
    def embeddings(self, embedding_function: object, model: str, credentials: object, project: str, region: str) -> object:
        """Embed the documents using the specified embedding function
        
        Parameters
        ----------
        embedding_function : object
            Embedding function to be used
        
        model : str
            Model name
        
        credentials : object
            GCP service account credentials
        
        project : str
            GCP project ID

        region : str
            GCP region

        Returns
        -------
        embedding_function : object
            Embedding function object

        """
        return embedding_function(model_name=model, credentials=credentials, project=project, location=region)
    
    def splittext(self, splitter: object, chunk_size: int, chunk_overlap: float, add_start_index: bool, documents: list) -> list:
        """Split the text into smaller chunks
        
        Parameters
        ----------
        splitter : object
            Text splitter object

        chunk_size : int
            Size of the chunks

        chunk_overlap : float
            Overlap between the chunks

        add_start_index : bool
            Whether to add the start index
        
        documents : list
            List of documents to be split

        Returns
        -------
        splits : list
            List of split documents

        """

        chunk_size = int(chunk_size)
        chunk_overlap = int(chunk_overlap)

        text_splitter = splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=add_start_index)
        return text_splitter.split_documents(documents)

    def run(self, input: str) -> str:
        """Main function to run the RAG application

        Parameters
        ----------
        input : str
            User input question

        Returns
        -------
        response : str
            Response from the RAG model

        """
        # Initialize VertexAI
        credentials = self.init_vertexai()

        # Collecting data from sources
        titles = ["Python (programming language)", "Java (programming language)", "C++", "JavaScript", "Rust (programming language)", "Go (programming language)", "Artificial Intelligence", "Machine Learning", 
                  "Natural Language Processing", "Deep Learning", "Reinforcement Learning", "Computer Vision", "Data Science", "Big Data", "Data Analytics", "Data Mining", "Data Warehousing", "Data Visualization"]
        
        documents = self.collecting_docs(titles)
        # print(f"Number of documents collected: {len(documents)}")

        # Initialize the embeddings
        embeddings = self.embeddings(VertexAIEmbeddings, "text-embedding-004", credentials, self.PROJECT_ID, self.REGION)

        # Split the documents into smaller chunks
        splits = self.splittext(RecursiveCharacterTextSplitter, 100, 50, True, documents)

        # Store it in a vector store
        chroma_store = Chroma.from_documents(splits, embedding=embeddings)

        # Initialize retriever
        retriever = chroma_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        # Initialize the llm model
        llm = ChatVertexAI(
            model_name="gemini-1.5-flash-001",
            temperature=0,
            max_tokens=None,
            max_retries=3,
            stop=None,
            credentials=credentials,
            project=self.PROJECT_ID,
            location=self.REGION,
            # other params...
        )

        # Initialize the prompt
        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful chatbot that can provide information about various topics. "
                    "You can answer questions, provide explanations, and give examples. "
                    "You can also ask questions to clarify the user's intent. "
                    "You can also provide links to relevant resources. "
                    "Use the following pieces of retrieved context to answer the user's question. "
                    "If you don't know the answer, say that you don't know. "
                    "Use three sentences maximum and keep the answer simple and concise. "
                    "\n\n"
                    "Context: {context}" 
                    
                ),
                (
                    "user",
                    "{input}"
                    
                )
            ]
        )

        # Initialize the chain
        rag_chain = (
            {
                "context": retriever | self.format_docs,
                "input": RunnablePassthrough(),
            }
            | prompt_template
            | llm
            | StrOutputParser()
        )

        # Run the chain
        response = rag_chain.invoke(input)
        print(response)
        return response

def main() -> None:
    """Main function which recursively asks the user for input and runs the RAG application
    
    Returns
    -------
    None

    """
    
    while True: 
        print("")
        user_input = input("Hi my name is raggy and I am your helpful bot which can answer any question regarding AI, ML or even Deep Learning! \n\n\
Enter a question you would like me to answer (or type 'exit' or enter '1' to quit): ")
        
        if user_input.lower() == "exit" or user_input == "1":
            print("")
            print("Exiting...")
            break
        response = rag_app().run(user_input)
        print("")
        print(response)
        print("\n")

if __name__ == "__main__":
    main()