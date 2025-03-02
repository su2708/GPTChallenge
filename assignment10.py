import streamlit as st
from openai import OpenAI
import os

# Streamlit 기본 설정
st.set_page_config(page_title="Research Assistant", layout="wide")

st.title("🔍 Research Assistant with OpenAI")
st.write("Ask anything, and I'll research using Wikipedia, DuckDuckGo, and web scraping!")

# 📌 Sidebar: OpenAI API Key 입력 필드
st.sidebar.header("🔑 OpenAI API Key")
api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
if not api_key:
    st.warning("⚠️ Please enter your OpenAI API Key in the sidebar!")
    st.stop()

# API 키 설정
os.environ["OPENAI_API_KEY"] = api_key
client = OpenAI(api_key=api_key)

# 📌 Sidebar: Github Repo 링크
st.sidebar.markdown("### 📝 [View on GitHub](https://github.com/su2708/GPTChallenge)")

# OpenAI Assistant 생성 (캐시 사용으로 중복 생성 방지)
@st.cache_resource
def create_assistant():
    return client.beta.assistants.create(
        name="Research Assistant",
        instructions=(
            "You are a research expert. Use Wikipedia, DuckDuckGo, and web scraping "
            "to provide detailed, well-organized answers with citations."
        ),
        tools=[{"type": "file_search"}],
        model="gpt-4o-mini",
    )

assistant = create_assistant()

# 세션 상태 초기화 (대화 기록 저장)
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! What do you want to research?"}]

# 채팅 UI 표시
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 사용자 입력
if user_query := st.chat_input("Ask your research question here..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # OpenAI Assistant 응답 생성
    thread = client.beta.threads.create(messages=[{"role": "user", "content": user_query}])
    run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant.id)

    with st.spinner("Researching..."):
        while run.status in ["queued", "in_progress"]:
            run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

        response = client.beta.threads.messages.list(thread_id=thread.id)
        assistant_reply = response.data[0].content[0].text.value

    # 답변 저장 및 표시
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
    with st.chat_message("assistant"):
        st.markdown(assistant_reply)
