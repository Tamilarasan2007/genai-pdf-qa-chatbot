## Development of a PDF-Based Question-Answering Chatbot Using LangChain

### AIM:
To design and implement a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain, and to evaluate its effectiveness by testing its responses to diverse queries derived from the document's content.

### PROBLEM STATEMENT:
To develop a PDF-based question-answering chatbot that allows users to upload PDF documents and ask questions about their content. The system uses LangChain, HuggingFace embeddings, FAISS vector database, and Groq LLM to retrieve relevant information and generate answers through an interactive Panel interface.

### DESIGN STEPS:

#### STEP 1:
Load the PDF document using PyPDFLoader and split its content into smaller chunks using RecursiveCharacterTextSplitter. Generate embeddings using HuggingFaceEmbeddings.

#### STEP 2:
Store the document embeddings in a FAISS vector database and create a retriever to find the top 4 relevant document chunks based on the user's question.

#### STEP 3:
Pass the retrieved context, chat history, and user question to the Groq language model using PromptTemplate. Generate answers and display them through the Panel GUI along with source documents.

### PROGRAM:
```python
import os
import panel as pn

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import PromptTemplate

pn.extension()
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=api_key
)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

retriever = None
chat_history = []

def load_db(file):

    global retriever

    loader = PyPDFLoader(file)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    docs = splitter.split_documents(documents)

    db = FAISS.from_documents(docs, embeddings)

    retriever = db.as_retriever(
        search_kwargs={"k": 4}
    )

    return len(docs)

def ask_question(question):

    global chat_history

    if retriever is None:
        return "Please upload a PDF first.", []

    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    history = ""

    for q, a in chat_history:
        history += f"User: {q}\nAssistant: {a}\n"

    template = """
    You are a PDF question-answering chatbot.

    Answer the question using only the provided context.

    If the answer is not available in the context,
    say "I don't know based on the provided PDF." and provide the information from your llm knowledge.

    Context:
    {context}

    Chat History:
    {history}

    Question:
    {question}

    Answer:
    """

    prompt = PromptTemplate(
        input_variables=["context", "history", "question"],
        template=template
    )

    formatted_prompt = prompt.format(
        context=context,
        history=history,
        question=question
    )

    response = llm.invoke(formatted_prompt)

    answer = response.content

    chat_history.append((question, answer))

    return answer, docs

# Panel GUI
title = pn.pane.Markdown(
    "# ",
    styles={"font-size": "0px"}
)

description = pn.pane.Markdown(
    "Upload pdf and ask qsn."
)

file_input = pn.widgets.FileInput(
    accept=".pdf"
)

load_button = pn.widgets.Button(
    name="Load PDF",
    button_type="primary"
)

status = pn.pane.Markdown("No PDF uploaded.")

question_input = pn.widgets.TextInput(
    placeholder="Enter qsn"
)

send_button = pn.widgets.Button(
    name="Send",
    button_type="success"
)

clear_button = pn.widgets.Button(
    name="Clear His",
    button_type="warning"
)

chat_display = pn.Column(
    sizing_mode="stretch_width"
)

source_display = pn.Column()


def load_pdf(event):

    global retriever, chat_history

    if file_input.value is None:
        status.object = "Please select a PDF file."
        return

    try:

        with open("uploaded.pdf", "wb") as f:
            f.write(file_input.value)

        status.object = "Loading PDF..."

        count = load_db("uploaded.pdf")

        chat_history = []
        chat_display.clear()
        source_display.clear()

        status.object = (
            f"PDF loaded successfully! "
            f"Total chunks: {count}"
        )

    except Exception as e:
        status.object = f"Error: {str(e)}"

def send_question(event):

    question = question_input.value.strip()

    if not question:
        return

    if retriever is None:
        status.object = "Please upload a PDF first."
        return

    try:

        answer, docs = ask_question(question)

        chat_display.append(
            pn.pane.Markdown(
                f"**You:** {question}"
            )
        )

        chat_display.append(
            pn.pane.Markdown(
                f"**Chatbot:** {answer}"
            )
        )

        source_display.clear()

        source_display.append(
            pn.pane.Markdown("### Source Documents")
        )

        for doc in docs:

            page = doc.metadata.get("page", "Unknown")

            source_display.append(
                pn.pane.Markdown(
                    f"**Page {page + 1 if isinstance(page, int) else page}:**\n"
                    f"{doc.page_content[:300]}..."
                )
            )

        question_input.value = ""

    except Exception as e:
        status.object = f"Error: {str(e)}"

def clear_history(event):

    global chat_history

    chat_history = []

    chat_display.clear()
    source_display.clear()

    status.object = "history cleared."

load_button.on_click(load_pdf)
send_button.on_click(send_question)
clear_button.on_click(clear_history)


# dashboard
app = pn.Column(

    title,

    description,

    pn.layout.Divider(),

    pn.pane.Markdown("### Upload PDF"),

    pn.Row(
        file_input,
        load_button
    ),

    status,

    pn.layout.Divider(),

    pn.pane.Markdown("### Conversation"),

    pn.Row(
        question_input,
        send_button
    ),

    clear_button,

    chat_display,

    pn.layout.Divider(),

    pn.Column(
        source_display,
        sizing_mode="stretch_width"
    ),

    width=850,
    sizing_mode="stretch_width"
)

app.servable()

pn.serve(app, show=True)
```

### OUTPUT:
<img width="1535" height="818" alt="image" src="https://github.com/user-attachments/assets/5cdc6716-e5e9-46cb-80eb-c38d7039db13" />


### RESULT:
Thus, a PDF-based question-answering chatbot was successfully developed using LangChain, HuggingFace embeddings, FAISS vector database, and Groq LLM. The application processes PDF documents, retrieves relevant information, generates answers to user queries, and displays the responses through an interactive Panel interface.
