# [translate:RAG 기반 PDF 분석 Assistant]

논문 등 PDF 파일을 쉽게 분석하고 이해할 수 있도록 도와주는 [translate:Retrieval-Augmented Generation(RAG)] 기반 Assistant입니다. 복잡한 수식, 이미지 등 논문의 주요 정보에 대해 인터랙티브하게 설명하고, 질의응답을 통해 누구나 논문을 손쉽게 분석할 수 있습니다.

## 프로젝트 소개
- **목적**: PDF로 구성된 논문과 연구자료의 내용을 쉽게 분석하여 누구나 이해할 수 있는 정보를 제공하는 도구입니다.
- **해결하는 문제**: 어렵고 복잡한 논문의 수식/이미지를 [translate:LLM] 기반으로 쉽게 설명하여, 비전문가도 논문을 분석하고 이해할 수 있습니다.

## 주요 기술
- upstage 기반 pdf 분석
- 이미지 및 방정식 llm 분석
- streamlit 활용
- FAISS db
- reOrder, reRank 통한 정확성 향상

## 설치 방법
1. 저장소를 클론합니다.
2. [Poetry]를 설치합니다 (https://python-poetry.org/docs/).
3. 프로젝트 폴더에서 아래 명령어를 실행하여 의존성을 설치합니다:> poetry install

## 사용 예시
- PDF 파일을 업로드하면 내용 요약, 수식 및 이미지를 별도 추출
- 사용자가 질문 입력 시, 논문 내용 기반으로 [translate:LLM] 분석 및 답변 제공
- .env copy 를 .env 로 name 변경 
- UPSTAGE_API_KEY,COHERE_API_KEY,LANGSMITH_TRACING,LANGSMITH_ENDPOINT,LANGSMITH_API_KEY,LANGSMITH_PROJECT,OPENAI_API_KEY 입력

## Sample Image
<img width="860" height="640" alt="화면 캡처 2025-10-27 124111" src="https://github.com/user-attachments/assets/e15b63d2-82f9-4810-bd9f-69616797a362" />


## 참고 자료
- [Streamlit 공식 문서](https://docs.streamlit.io/)
- [LangChain RAG 샘플](https://github.com/streamlit/example-app-langchain-rag)


