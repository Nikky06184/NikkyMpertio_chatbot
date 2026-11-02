from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from dotenv import load_dotenv

import os
import hmac
import hashlib
import asyncio
import logging
import httpx


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

WHATSAPP_ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")

WHATSAPP_API_VERSION = os.getenv(
    "WHATSAPP_API_VERSION",
    "v25.0"
)

WEBHOOK_VERIFY_TOKEN = os.getenv("WEBHOOK_VERIFY_TOKEN")

META_APP_SECRET = os.getenv("META_APP_SECRET")


# ============================================================
# 2. LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# 3. VALIDATE ENVIRONMENT VARIABLES
# ============================================================
required_variables = {
    "GROQ_API_KEY": GROQ_API_KEY,
    "WHATSAPP_ACCESS_TOKEN": WHATSAPP_ACCESS_TOKEN,
    "WHATSAPP_PHONE_NUMBER_ID": WHATSAPP_PHONE_NUMBER_ID,
    "WEBHOOK_VERIFY_TOKEN": WEBHOOK_VERIFY_TOKEN,
    "META_APP_SECRET": META_APP_SECRET,
}
missing_variables = [
    key
    for key, value in required_variables.items()
    if not value
]

if missing_variables:
    raise RuntimeError(
        f"Missing environment variables: {missing_variables}"
    )


# ============================================================
# 4. LANGCHAIN IMPORTS
# ============================================================

from langchain_groq import ChatGroq

from langchain_community.embeddings import (
    HuggingFaceEmbeddings
)

from langchain_community.vectorstores import FAISS


# ============================================================
# 5. CREATE FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Mperito WhatsApp AI Chatbot",
    version="3.0"
)


# ============================================================
# 6. LOAD EMBEDDING MODEL
# ============================================================

logger.info("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

logger.info("Embedding model loaded successfully.")


# ============================================================
# 7. LOAD FAISS VECTOR DATABASE
# ============================================================

logger.info("Loading FAISS Vector Database...")

vector_db = FAISS.load_local(
    "vectormoretxtfastapi",
    embeddings,
    allow_dangerous_deserialization=True
)

logger.info("FAISS Vector Database loaded successfully.")


# ============================================================
# 8. CREATE GROQ LLM
# ============================================================

logger.info("Loading Groq LLM...")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    groq_api_key=GROQ_API_KEY,
    temperature=0
)

logger.info("Groq LLM loaded successfully.")
logger.info("Mperito WhatsApp Chatbot is ready.")


# ============================================================
# 9. REQUEST MODEL FOR /chat3 TESTING
# ============================================================

class UserQuery(BaseModel):
    user_query: str


# ============================================================
# 10. HOME ENDPOINT
# ============================================================

@app.get("/")
async def home():

    return {
        "status": "online",
        "message": "Mperito WhatsApp AI Chatbot is running"
    }


# ============================================================
# 11. NORMAL CHATBOT TEST ENDPOINT
# ============================================================

@app.post("/chat3")
async def chat(data: UserQuery):

    answer = await asyncio.to_thread(
        get_chatbot_answer,
        data.user_query
    )

    return {
        "question": data.user_query,
        "answer": answer
    }


# ============================================================
# 12. GREETING HANDLER
# ============================================================

def is_greeting(message: str) -> bool:

    greeting_words = {
        "hi",
        "hello",
        "hey",
        "hii",
        "hiii",
        "good morning",
        "good afternoon",
        "good evening",
        "good night"
    }

    cleaned_message = message.lower().strip()

    return cleaned_message in greeting_words


def get_greeting_response() -> str:

    return (
        "Hello! 👋\n"
        "Welcome to Mperito.\n"
        "How can I assist you today?"
    )


# ============================================================
# 13. RAG CHATBOT FUNCTION
# ============================================================

def get_chatbot_answer(query: str) -> str:

    logger.info(
        "Running RAG search for customer question: '%s'",
        query
    )

    # --------------------------------------------------------
    # Greeting
    # --------------------------------------------------------

    if is_greeting(query):

        return get_greeting_response()


    # --------------------------------------------------------
    # Search FAISS
    # --------------------------------------------------------

    documents_chunk = vector_db.similarity_search(
        query,
        k=3
    )

    if not documents_chunk:

        return "Information not available in the document."


    # --------------------------------------------------------
    # Create context
    # --------------------------------------------------------

    context = "\n\n".join(
        doc.page_content
        for doc in documents_chunk
    )


    # --------------------------------------------------------
    # RAG Prompt
    # --------------------------------------------------------

    prompt = f"""
You are Mperito's AI customer support assistant.

Your job is to answer the customer's question using ONLY
the information contained in the Context.

STRICT RULES:

1. Use ONLY the Context below.
2. Do NOT use your general knowledge.
3. Do NOT guess.
4. Do NOT invent prices, products, phone numbers,
   addresses, websites, offers, or company information.
5. If the answer is not present in the Context, reply exactly:

Information not available in the document.

6. Keep the answer short and useful.
7. Answer in bullet points.
8. Each bullet should be a short sentence.
9. Do not mention FAISS, LangChain, Groq, embeddings,
   vector database, or internal implementation.
10. Answer as a customer-facing Mperito assistant.

Context:
--------------------
{context}
--------------------

Customer Question:
{query}

Answer:
"""


    # --------------------------------------------------------
    # Call Groq
    # --------------------------------------------------------

    try:

        response = llm.invoke(prompt)

        answer = response.content.strip()

        logger.info(
            "Generated chatbot answer: %s",
            answer
        )

        return answer

    except Exception as e:

        logger.exception(
            "Groq error while generating answer"
        )

        return (
            "Sorry, I could not process your request "
            "right now."
        )


# ============================================================
# 14. META WEBHOOK VERIFICATION
# ============================================================

@app.get("/webhook")
async def verify_webhook(request: Request):

    params = request.query_params

    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    logger.info(
        "Meta webhook verification request received."
    )

    if (
        mode == "subscribe"
        and token == WEBHOOK_VERIFY_TOKEN
        and challenge
    ):

        logger.info(
            "Webhook verification successful."
        )

        return PlainTextResponse(
            content=challenge,
            status_code=200
        )


    logger.error(
        "Webhook verification failed."
    )

    raise HTTPException(
        status_code=403,
        detail="Verification failed"
    )


# ============================================================
# 15. VERIFY META SIGNATURE
# ============================================================

async def verify_meta_signature(
    request: Request,
    body: bytes
):

    signature = request.headers.get(
        "X-Hub-Signature-256"
    )


    if not signature:

        logger.error(
            "Missing X-Hub-Signature-256 header."
        )

        raise HTTPException(
            status_code=401,
            detail="Missing signature"
        )


    secret = META_APP_SECRET.strip().encode(
        "utf-8"
    )


    generated_hash = hmac.new(
        secret,
        body,
        hashlib.sha256
    ).hexdigest()


    expected_signature = (
        f"sha256={generated_hash}"
    )


    if not hmac.compare_digest(
        signature,
        expected_signature
    ):

        logger.error(
            "Meta signature verification failed."
        )

        raise HTTPException(
            status_code=401,
            detail="Invalid signature"
        )


    logger.info(
        "Meta signature verification successful."
    )


# ============================================================
# 16. SEND MESSAGE TO WHATSAPP
# ============================================================

async def send_whatsapp_message(
    recipient_phone: str,
    message: str
):

    url = (
        f"https://graph.facebook.com/"
        f"{WHATSAPP_API_VERSION}/"
        f"{WHATSAPP_PHONE_NUMBER_ID}/messages"
    )


    headers = {
        "Authorization": (
            f"Bearer {WHATSAPP_ACCESS_TOKEN}"
        ),
        "Content-Type": "application/json"
    }


    payload = {

        "messaging_product": "whatsapp",

        "recipient_type": "individual",

        "to": recipient_phone,

        "type": "text",

        "text": {
            "preview_url": False,
            "body": message
        }
    }


    try:

        async with httpx.AsyncClient(
            timeout=30
        ) as client:

            response = await client.post(
                url,
                headers=headers,
                json=payload
            )


        logger.info(
            "WhatsApp API status: %s",
            response.status_code
        )


        if response.status_code >= 400:

            logger.error(
                "WhatsApp API error: %s",
                response.text
            )

            return False


        logger.info(
            "WhatsApp message sent successfully: %s",
            response.text
        )

        return True


    except Exception as e:

        logger.exception(
            "Error sending WhatsApp message"
        )

        return False


# ============================================================
# 17. PROCESS CUSTOMER MESSAGE
# ============================================================

async def process_whatsapp_message(
    sender_phone: str,
    user_message: str
):

    logger.info(
        "Processing customer message from %s",
        sender_phone
    )

    logger.info(
        "Customer message: '%s'",
        user_message
    )


    try:

        # ----------------------------------------------------
        # Run RAG in background thread
        # ----------------------------------------------------

        answer = await asyncio.to_thread(
            get_chatbot_answer,
            user_message
        )


        logger.info(
            "Generated answer: '%s'",
            answer
        )


        # ----------------------------------------------------
        # Send answer to WhatsApp
        # ----------------------------------------------------

        success = await send_whatsapp_message(
            sender_phone,
            answer
        )


        if success:

            logger.info(
                "Response delivered to WhatsApp API."
            )

        else:

            logger.error(
                "Could not send response to WhatsApp."
            )


    except Exception as e:

        logger.exception(
            "Error processing WhatsApp message."
        )


        await send_whatsapp_message(
            sender_phone,
            "Sorry, I could not process your request."
        )


# ============================================================
# 18. WHATSAPP WEBHOOK
# ============================================================

@app.post("/webhook")
async def whatsapp_webhook(
    request: Request,
    background_tasks: BackgroundTasks
):

    # --------------------------------------------------------
    # Read raw request body
    # --------------------------------------------------------

    body = await request.body()


    # --------------------------------------------------------
    # Verify Meta request
    # --------------------------------------------------------

    await verify_meta_signature(
        request,
        body
    )


    # --------------------------------------------------------
    # Convert JSON
    # --------------------------------------------------------

    try:

        data = await request.json()

    except Exception:

        logger.error(
            "Invalid JSON received from Meta."
        )

        return {
            "status": "invalid_json"
        }


    logger.info(
        "WhatsApp webhook event received."
    )


    # --------------------------------------------------------
    # Basic WhatsApp webhook structure
    # --------------------------------------------------------

    try:

        entries = data.get(
            "entry",
            []
        )


        for entry in entries:

            changes = entry.get(
                "changes",
                []
            )


            for change in changes:

                value = change.get(
                    "value",
                    {}
                )


                # =================================================
                # CUSTOMER MESSAGES
                # =================================================

                messages = value.get(
                    "messages",
                    []
                )


                for message in messages:

                    message_type = message.get(
                        "type"
                    )


                    sender_phone = message.get(
                        "from"
                    )


                    message_id = message.get(
                        "id"
                    )


                    logger.info(
                        "Message received from %s; "
                        "type=%s; message_id=%s",
                        sender_phone,
                        message_type,
                        message_id
                    )


                    # ------------------------------------------------
                    # Only process TEXT messages
                    # ------------------------------------------------

                    if message_type != "text":

                        logger.info(
                            "Ignoring non-text message type: %s",
                            message_type
                        )

                        continue


                    text_data = message.get(
                        "text",
                        {}
                    )


                    user_message = text_data.get(
                        "body",
                        ""
                    ).strip()


                    if not user_message:

                        logger.info(
                            "Empty customer message received."
                        )

                        continue


                    logger.info(
                        "Customer message: '%s'",
                        user_message
                    )


                    logger.info(
                        "Character count: %s",
                        len(user_message)
                    )


                    # ------------------------------------------------
                    # Process in background
                    # ------------------------------------------------

                    background_tasks.add_task(
                        process_whatsapp_message,
                        sender_phone,
                        user_message
                    )


                # =================================================
                # WHATSAPP STATUS UPDATES
                # =================================================

                statuses = value.get(
                    "statuses",
                    []
                )


                for status in statuses:

                    message_id = status.get(
                        "id"
                    )

                    status_value = status.get(
                        "status"
                    )

                    recipient = status.get(
                        "recipient_id"
                    )


                    logger.info(
                        "WhatsApp status: "
                        "message_id=%s, "
                        "status=%s, "
                        "recipient=%s",
                        message_id,
                        status_value,
                        recipient
                    )


    except Exception as e:

        logger.exception(
            "Error parsing WhatsApp webhook."
        )


    # --------------------------------------------------------
    # IMPORTANT:
    # Always return 200 after accepting webhook
    # --------------------------------------------------------

    return {
        "status": "received"
    }

