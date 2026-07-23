from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

print("Loading environment variables...")
load_dotenv()

# -------------------------------------
# Create FastAPI App
# -------------------------------------

app = FastAPI(
    title="Mperito Chatbot API",
    version="1.0"
)

# -------------------------------------
# Load Embedding Model
# -------------------------------------

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------------------
# Load Vector Database
# -------------------------------------

print("Loading FAISS Vector Database...")

vector_db = FAISS.load_local(
    "vectormoretxtfastapi",
    embeddings,
    allow_dangerous_deserialization=True
)

# -------------------------------------
# Create Groq LLM
# -------------------------------------

print("Loading Groq LLM...")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

print("Chatbot Ready")

# -------------------------------------
# Request Model

# -------------------------------------

class UserQuery(BaseModel):
    user_query: str

# Home Endpoint
#  -------------------------------------
#-------------------------------------

@app.get("/")
def home():
    return {
        "message": "Welcome to Mperito Chatbot API",
        "hello":"   hello how can i help you "
    }
###############################################

# -------------------------------------
# Chat Endpoint
# -------------------------------------

@app.post("/chat3")
def chat(data: UserQuery):

    query = data.user_query

    # Retrieve relevant documents
    documents_chunk = vector_db.similarity_search(
        query,
        k=3
    )

    # Create context
    
    context = "\n\n".join(
        doc.page_content
        for doc in documents_chunk
    )
    


    prompt = f"""
        You are Mperito's AI Assistant.

    Your task is to answer the user's question ONLY using the information provided in the Context.

    STRICT RULES (Follow every rule exactly):

    1. Use ONLY the Context provided below.
    2. Never use your own knowledge.
    3. Never guess, assume, or make up information.
    4. If the answer is not present in the Context, reply exactly:
    "Information not available in the document."
    5. Keep the answer short and direct.
    6. Answer in bullet points only.
    7. Each bullet should contain only one short sentence.
    8. Do not write explanations, introductions, summaries, or conclusions.
    9. Do not repeat the same information.
    10. Preserve names, numbers, dates, email addresses, website URLs, and phone numbers exactly as they appear in the Context.
    11. If the user asks for:
        - Contact number → Return only the phone number(s).
        - Email → Return only the email address(es).
        - Website → Return only the website link(s).
        - Address → Return only the address.
    12. If multiple answers exist in the Context, list all of them.
    13. Do not mention that you are using the Context.
    14. Do not generate Markdown headings.
    15. If the Context contains a URL, include it exactly as written.
    16. If the Context contains a mobile number, include it exactly as written.
    17. If the answer is a single value (name, phone, email, website, price, date), return only that value as a bullet point.
    18. Keep the total answer under 100 words unless the Context itself requires more.
    19. Maintain the same language as the user's question. If the question is in Hindi, answer in Hindi. If it is in English, answer in English.

    Context:
    {context}

    Question:

    {query}
    """

    response = llm.invoke(prompt)

    return {
        "question": query,
        "answer": response.content
    }


