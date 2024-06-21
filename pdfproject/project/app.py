# Import necessary modules
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import cassio

# Initialize connections and set tokens
ASTRA_DB_APPLICATION_TOKEN = "AstraCS:jTgMrBvQeGUOhwsXdtDzjZKi:603f6da4d817cca4ecc081d30c8f901ad6db715c1dd38ae93c6724e438eb5f1c"
ASTRA_DB_ID = "cbcd4500-e7bc-482d-a040-d7615c6fe6cf"
OPENAI_API_KEY = "open_ai_ki"


# Read the PDF file
pdf_path = "C:\\Users\\udayb\\OneDrive\\Desktop\\AntiraggingAffidavitForm.pdf"
pdfreader = PdfReader(pdf_path)
raw_text = ''
for page in pdfreader.pages:
    content = page.extract_text()
    if content:
        raw_text += content

# Initialize CassIO connection
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

# Initialize OpenAI LLM and embeddings
llm = OpenAI(openai_api_key=OPENAI_API_KEY)
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Initialize Cassandra vector store
astra_vector_store = Cassandra(
    embedding=embedding,
    table_name="qa_mini_demo",
    session=None,
    keyspace=None,
)

# Split text using CharacterTextSplitter
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
)
texts = text_splitter.split_text(raw_text)

# Add texts to the vector store
astra_vector_store.add_texts(texts[:50])
print("Inserted %i headlines." % len(texts[:50]))

# Create an index wrapper
astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)

# Interactive Q&A loop
first_question = True
while True:
    if first_question:
        query_text = input("\nEnter your question (or type 'quit' to exit): ").strip()
    else:
        query_text = input("\nWhat's your next question (or type 'quit' to exit): ").strip()

    if query_text.lower() == "quit":
        break

    if query_text == "":
        continue

    first_question = False

    print("\nQUESTION: \"%s\"" % query_text)
    answer = astra_vector_index.query(query_text, llm=llm).strip()
    print("ANSWER: \"%s\"\n" % answer)

    print("FIRST DOCUMENTS BY RELEVANCE:")
    for doc, score in astra_vector_store.similarity_search_with_score(query_text, k=4):
        print("    [%0.4f] \"%s ...\"" % (score, doc.page_content[:84]))
