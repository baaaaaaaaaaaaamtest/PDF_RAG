import os
import streamlit as st
from langchain_teddynote import logging
from dotenv import load_dotenv
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_teddynote.prompts import load_prompt
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from modules.utils import *

load_dotenv()

logging.langsmith("[Project] Upstage PDF RAG")


# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")

# 파일 업로드 전용 폴더
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")


st.title("PDF 기반 QA💬")

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성한다.
    st.session_state["messages"] = []

if "llm" not in st.session_state:
    # 아무런 파일을 업로드 하지 않을 경우
    st.session_state["llm"] = None

# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")

    # 파일 업로드
    uploaded_file = st.file_uploader("파일 업로드", type=["pdf"])

    # uploaded_btn_apply = st.button("pdf 분석")

    # 모델 선택 메뉴
    selected_model = st.selectbox(
        "LLM 선택", ["gpt-4o", "gpt-4-turbo", "gpt-4o-mini"], index=2
    )

    selected_method = st.selectbox("분석 방법 선택", ["normal", "upstage"], index=1)


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 파일을 캐시 저장(시간이 오래 걸리는 작업을 처리할 예정)
# @st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...")
def embed_file(file):
    print(file)
    # 업로드한 파일을 캐시 디렉토리에 저장합니다.
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    output_folder = os.path.splitext(file_path)[0]
    print(file_path)
    with open(file_path, "wb") as f:
        f.write(file_content)

    # 분석 방법 선택
    if selected_method == "normal":
        print("normal")
        retriever = noraml_retrievr(file_path)
    elif selected_method == "upstage":
        print("upstage")
        final_output_path = f"{output_folder}\\analysis.html"
        if os.path.isfile(final_output_path):
            print("File exists.")
        else:
            print("File does not exists.")
            split_files = split_pdf(file_path)
            analyser = LayoutAnalyzer(os.environ.get("UPSTAGE_API_KEY"))
            analyser.execute(split_files)  # json 파일로 변환하여 저장 완료
            image_processor = PDFImageProcessor(file_path)
            html_content = image_processor.extract_images()
            add_summary_to_html(html_content, image_processor.output_folder)
            print("변환 완료")
        docs = get_docs(output_folder)
        faiss_retriever = get_retriever(docs)
        bm25_retriever = get_bm25_retriever(docs)
        retriever = get_esenmble_retriever(faiss_retriever, bm25_retriever)

    return retriever


# 체인 생성
def create_chain(model_name="gpt-4o"):
    # 단계 7: 언어모델(LLM) 생성
    # 모델(LLM) 을 생성합니다.
    llm = ChatOpenAI(model_name=model_name, temperature=0)
    return llm


#  업로드 되었을 때
if uploaded_file:
    print(selected_method)
    print(selected_model)

    # 파일 업로드 후 retriever 생성 (작업시간이 오래 걸릴 예정...)
    retriever = embed_file(uploaded_file)
    st.session_state["retriever"] = retriever
    llm = create_chain(model_name=selected_model)
    st.session_state["llm"] = llm
    print("uploaded_file 완료")

# 초기화 버튼이 눌리면...
if clear_btn:
    st.session_state["messages"] = []


# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 경고 메시지를 띄우기 위한 빈 영역
warning_msg = st.empty()

# 만약에 사용자 입력이 들어오면...
if user_input:
    # chain 을 생성
    llm = st.session_state["llm"]
    if llm is not None:
        # 사용자의 입력
        st.chat_message("user").write(user_input)
        # 스트리밍 호출
        # 단계 6: 프롬프트 생성(Create Prompt)
        # 프롬프트를 생성합니다.
        prompt = load_prompt("prompts/pdf-rag.yaml", encoding="utf-8")
        retriever = st.session_state["retriever"]
        reranker_docs = get_reranker(retriever, user_input)
        reorder_docs = reorder_documents(reranker_docs)
        # print(reorder_docs)
        # print(user_input)
        chain = prompt | llm | StrOutputParser()
        response = chain.stream({"question": user_input, "context": reorder_docs})
        with st.chat_message("assistant"):
            # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
            container = st.empty()

            ai_answer = ""
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

        # 대화기록을 저장한다.
        add_message("user", user_input)
        add_message("assistant", ai_answer)
    else:
        # 파일을 업로드 하라는 경고 메시지 출력
        warning_msg.error("파일을 업로드 해주세요.")

# 일반 RAG 로그 https://smith.langchain.com/public/b872dd3f-da9a-41ec-a3fd-4e72a8ceb2cd/r
# upstage 활용 이미지 크롭 및 summary 적용 결과 https://smith.langchain.com/public/0f3ac8bb-6743-4c97-a598-d1497a593f3e/r
