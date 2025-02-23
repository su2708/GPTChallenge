import json
import streamlit as st

from langchain_unstructured import UnstructuredLoader
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import BaseOutputParser
from pathlib import Path

st.set_page_config(
    page_title="Assignment #7 QuizGPT",
    page_icon="❓",
)

st.title("Quiz GPT")

# 사용자 OpenAI API Key 입력 
st.sidebar.header("🔑 OpenAI API Key")
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

if not api_key:
    st.warning("OpenAI API Key를 입력하세요.")
    st.stop()

class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)


output_parser = JsonOutputParser()

llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    openai_api_key=api_key,
    callbacks=[
        StreamingStdOutCallbackHandler(),
    ],
)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

questions_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a helpful assistant that is role playing as a teacher.
        
        Based ONLY on the following context make 10 questions to test the user's knowledge about the text.
        
        You MUST consider the difficulty level.
        
        Each question should have 4 answers, three of them must be incorrect and one should be correct.
        
        Use (o) to signal the correct answer.
        
        Question examples:
        
        Question: What is the color of the ocean?
        Answers: Red | Yellow | Green | Blue(o)
        
        Question: What is the capital of Georgia?
        Answers: Baky | Tbilisi(o) | Manila | Beirut
        
        Question: When was Avatar released?
        Answers: 2007 | 2001 | 2009(o) | 1998
        
        Question: Who was Julius Caesar?
        Answers: A Roman Emperor(o) | Painter | Actor | Model
        
        Your turn!
        
        Context: {context}
        Difficulty: {difficulty}
        """
    )
])

questions_chain = {
    "context": format_docs
} | questions_prompt | llm

formatting_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a powerful formatting algorithm.

        You format exam questions into JSON format.
        Answers with (o) are the correct ones.

        Example Input:
        Question: What is the color of the ocean?
        Answers: Red|Yellow|Green|Blue(o)

        Question: What is the capital or Georgia?
        Answers: Baku|Tbilisi(o)|Manila|Beirut

        Question: When was Avatar released?
        Answers: 2007|2001|2009(o)|1998

        Question: Who was Julius Caesar?
        Answers: A Roman Emperor(o)|Painter|Actor|Model


        Example Output:

        ```json
        {{ "questions": [
                {{
                    "question": "What is the color of the ocean?",
                    "answers": [
                        {{
                            "answer": "Red",
                            "correct": false
                        }},
                        {{
                            "answer": "Yellow",
                            "correct": false
                        }},
                        {{
                            "answer": "Green",
                            "correct": false
                        }},
                        {{
                            "answer": "Blue",
                            "correct": true
                        }},
                    ]
                }},
                {{
                    "question": "What is the capital or Georgia?",
                    "answers": [
                        {{
                            "answer": "Baku",
                            "correct": false
                        }},
                        {{
                            "answer": "Tbilisi",
                            "correct": true
                        }},
                        {{
                            "answer": "Manila",
                            "correct": false
                        }},
                        {{
                            "answer": "Beirut",
                            "correct": false
                        }},
                    ]
                }},
                {{
                    "question": "When was Avatar released?",
                    "answers": [
                        {{
                            "answer": "2007",
                            "correct": false
                        }},
                        {{
                            "answer": "2001",
                            "correct": false
                        }},
                        {{
                            "answer": "2009",
                            "correct": true
                        }},
                        {{
                            "answer": "1998",
                            "correct": false
                        }},
                    ]
                }},
                {{
                    "question": "Who was Julius Caesar?",
                    "answers": [
                        {{
                            "answer": "A Roman Emperor",
                            "correct": true
                        }},
                        {{
                            "answer": "Painter",
                            "correct": false
                        }},
                        {{
                            "answer": "Actor",
                            "correct": false
                        }},
                        {{
                            "answer": "Model",
                            "correct": false
                        }},
                    ]
                }}
            ]
        }}
        ```
        Your turn!
        Questions: {context}
        """,
    )
])

formatting_chain = formatting_prompt | llm

if "messages" not in st.session_state:
    st.session_state["messages"] = []

@st.cache_resource(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./files/{file.name}"
    with open(file_path, "wb+") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredLoader(file_path)
    docs = splitter.split_documents(loader.load())
    return docs

@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, difficulty):
    chain = (
        RunnableLambda(lambda docs: {"context": format_docs(docs), "difficulty": difficulty})
        | questions_prompt
        | llm
        | formatting_chain
        | output_parser
    )
    return chain.invoke(_docs)

with st.sidebar:
    docs = None
    file = None
    
    # Level bar for difficulty selection using a select slider.
    difficulty = st.select_slider(
        "Select quiz difficulty",
        options=["Super Easy", "Easy", "Normal", "Hard", "Very Hard"],
        value="Normal",
    )

    # 파일 선택
    file = st.file_uploader(
        "Upload a. txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )
    if file:
        docs = split_file(file)

if not docs:
    st.markdown(
        """
        Welcome to QuizGPT!
        
        I will make a quiz from a file you upload!
        Use this chatbot to test your knowledge and help you study!

        1. Input your OpenAI API Key on the sidebar
        2. Adjust level.
        3. Upload your file on the sidebar.
        4. Get started!
        """
    )
else:
    response = run_quiz_chain(docs, difficulty)
    with st.form("questions_form"):
        # Dictionary to store each question's response.
        answers = {}
        for idx, question in enumerate(response["questions"]):
            st.write(question["question"])
            answers[idx] = st.radio(
                "Select an option",
                [answer["answer"] for answer in question["answers"]],
                index=None,
                key=f"question_radio_{idx}"
            )
        button = st.form_submit_button("Submit")

    if button:
        all_correct = True
        # Check each answer after submission.
        for idx, question in enumerate(response["questions"]):
            # Determine if the selected answer is correct.
            if {"answer": answers[idx], "correct": True} in question["answers"]:
                st.success(f"Question {idx+1} is correct!")
            else:
                st.error(f"Question {idx+1} is wrong!")
                all_correct = False

        # If every answer is correct, display balloons.
        if all_correct:
            st.balloons()