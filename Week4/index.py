from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_mistralai import MistralAIEmbeddings

load_dotenv()

# load document
loader = PyPDFLoader('innvonix_hr_policy.pdf')
docs = loader.load()
print(docs[0].metadata)

# split document in to chunks using RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
chunks = splitter.split_documents(docs)

# Create embeddings and the vector store  using HuggingFaceEmbeddings and FAISS
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings = MistralAIEmbeddings(
    model="mistral-embed",
    api_key=os.getenv("MISTRAL_API_KEY")
)
vector_store = FAISS.from_documents(chunks, embeddings)

# Turn the store into a retriever
retriever=vector_store.as_retriever(search_kwargs={"k": 3})

# prompt template
prompt = ChatPromptTemplate.from_template("""Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know."
Context: {context}
Question: {question}""")

# create LLM model using ChatGroq
llm = ChatGroq(
    model_name="openai/gpt-oss-120b", 
    temperature=0.3, 
    api_key=os.getenv("GROQ_API_KEY").strip()
)
# llm = ChatOllama(
#     model="qwen2.5:0.5b",
#     temperature=0.3
# )


# creating chain and formating data
def FormatDocs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    { "context": retriever | FormatDocs, "question": RunnablePassthrough() } | prompt | llm | StrOutputParser()
)

#function to display chunks and similariti score
def show_retrieved_chunks(question, k = 1):
    results = vector_store.similarity_search_with_score(question, k=k)
    print("\n" + "=" * 60)
    print("       RETRIEVED CHUNKS PASSED TO LLM")
    print("=" * 60)

    for i, (doc, score) in enumerate(results):
        print(f"\n📄 Chunk {i + 1}  |  Score: {score:.4f}  (lower = more relevant)")
        print(f"   Source : {doc.metadata.get('source', 'N/A')}")
        print(f"   Page   : {doc.metadata.get('page', 'N/A')}")
        # print(f"   Content:\n   {doc.page_content}")
        print("-" * 60)


# ask question
while True:
    question = input("Ask a question: ")
    if question.lower() == "exit":
        break

    # Show what chunks + scores the retriever found
    show_retrieved_chunks(question)

    # invoke rag chain
    answer = rag_chain.invoke(question)
    print(answer)
