import os
import pymupdf
from glob import glob
import json
import requests
from PIL import Image
from langchain_openai import ChatOpenAI
from langchain_text_splitters import HTMLHeaderTextSplitter
from langchain.schema import Document
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import cohere
from langchain_community.document_transformers import LongContextReorder
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from bs4 import BeautifulSoup
from markdownify import markdownify as markdown
from langchain_teddynote.models import MultiModal
from dotenv import load_dotenv
import base64
from modules.prompt import *

load_dotenv()


def get_gpt(model: str = "gpt-4.1-mini", temperature: int = 0):
    return ChatOpenAI(model=model, temperature=temperature)


def noraml_retrievr(file_path):
    """pdf load 활용, 전처리 없음"""
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()

    # 단계 2: 문서 분할(Split Documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)

    # 단계 3: 임베딩(Embedding) 생성
    embeddings = OpenAIEmbeddings()

    # 단계 4: DB 생성(Create DB) 및 저장
    # 벡터스토어를 생성합니다.
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    # 단계 5: 검색기(Retriever) 생성
    # 문서에 포함되어 있는 정보를 검색하고 생성합니다.
    retriever = vectorstore.as_retriever()
    return retriever


def split_pdf(filepath, batch_size=5):
    """
    입력 PDF를 분할 PDF 파일로 분할
    """
    # PDF 파일 열기
    basename = os.path.splitext(filepath)[0]
    filename = os.path.splitext(os.path.basename(basename))[0]
    os.makedirs(basename, exist_ok=True)
    input_pdf = pymupdf.open(filepath)
    num_pages = len(input_pdf)

    ret = []
    # PDF 분할
    for start_page in range(0, num_pages, batch_size):
        end_page = min(start_page + batch_size, num_pages) - 1
        # 분할된 PDF 저장
        # /folder/example.pdf = > ['/folder/example','.pdf']
        output_file = f"{basename}\\{filename}_{start_page:04d}_{end_page:04d}.pdf"
        print(f"분할 PDF 생성: {output_file}")
        with pymupdf.open() as output_pdf:
            output_pdf.insert_pdf(input_pdf, from_page=start_page, to_page=end_page)
            output_pdf.save(output_file)
            ret.append(output_file)

    # 입력 PDF 파일 닫기
    input_pdf.close()

    return ret, basename


def upstage_layout_analysis(input_files):
    analyzed_files = []
    config = {
        "ocr": False,
        "coordinates": True,
        "output_formats": "['html', 'text', 'markdown']",
        "model": "document-parse",
        "base64_encoding": "['figure', 'chart', 'table']",
    }
    for file in input_files:
        response = requests.post(
            "https://api.upstage.ai/v1/document-ai/document-parse",
            data=config,
            headers={"Authorization": f"Bearer {os.environ.get('UPSTAGE_API_KEY')}"},
            files={"document": open(file, "rb")},
        )

        # 응답 저장
        if response.status_code == 200:
            output_file = os.path.splitext(file)[0] + ".json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(response.json(), f, ensure_ascii=False)
            analyzed_files.append(output_file)
        else:
            raise ValueError(f"예상치 못한 상태 코드: {response.status_code}")
    return analyzed_files


def reOrder_id(analysis_file_path):
    """
    element 구성 변경
    id 정렬
    html 값 변경
    page 값 수정
    """
    element_content = []
    page_range = 5
    last_number = 0
    for i, path in enumerate(sorted(analysis_file_path)):
        with open(path, "r", encoding="utf-8") as file:
            # 파일 내용을 파이썬 객체로 변환
            data = json.load(file)  # 보통 JSON 배열이면 list 타입이 됨
            change_page = page_range * i
            for j, element in enumerate(data["elements"]):
                element["id"] = last_number + j
                html_content = element["content"]["html"]
                soup = BeautifulSoup(html_content, "html.parser")
                tag = soup.find(attrs={"id": True})
                tag["id"] = last_number + j
                element["content"]["html"] = str(tag)
                element["page"] = change_page + element["page"]
            last_number = len(data["elements"])
            element_content.extend(data["elements"])
    return element_content


def extract_image(element_content, basename):
    """
    element_content 에서 base64 정보 가져와 이미지로 저장
    """
    arr = []
    for idx, item in enumerate(element_content):
        _dict = dict()
        if "base64_encoding" in item:
            _dict = {"base64_encoding": item["base64_encoding"]}
            html_str = item["content"]["html"]
            soup = BeautifulSoup(html_str, "html.parser")
            img_tags = soup.find_all("img")
            if img_tags:
                base64_str = item["base64_encoding"]
                # base64 문자열 디코딩
                image_data = base64.b64decode(base64_str)
                image_path = os.path.join(f"{basename}\\image_{idx}.png")
                for img_tag in soup.find_all("img"):
                    # 기존 속성 모두 제거
                    img_tag.attrs.clear()
                    # 원하는 텍스트를 src 속성에 넣기
                    img_tag["src"] = image_path
                # 이미지 파일 저장
                _dict["content_text"] = str(img_tag)
                with open(f"{image_path}", "wb") as img_file:
                    img_file.write(image_data)
                print(f"Saved image: {image_path}")
            else:
                _dict["content_text"] = item["content"]["html"]
                print("이미지 태그가 없습니다.")
        else:
            _dict["content_text"] = item["content"]["html"]
            print("base64_encoding params 없습니다.")

        _dict["metadata"] = {"id": item["id"], "page": item["page"]}
        arr.append(_dict)
    return arr


def order_image_prompt(_element_content):
    """
    이미지를 LLM에 전달하기전 프롬프트, 이미지주소 정렬
    """
    image_urls: list[str] = []
    system_prompts: list[str] = []
    user_prompts: list[str] = []
    for idx, item in enumerate(_element_content):
        previous_context = (
            _element_content[idx - 1]["content_text"] if idx - 1 >= 0 else None
        )
        next_context = (
            _element_content[idx + 1]["content_text"]
            if idx + 1 < len(_element_content)
            else None
        )
        id_str = item["metadata"]["id"]
        html_str = item["content_text"]
        soup = BeautifulSoup(html_str, "html.parser")
        img_tag = soup.find("img")
        if img_tag:
            src_path = img_tag.get("src")
            image_urls.append(src_path)
            system_prompts.append(get_prompt_system_image_summary())
            user_prompts.append(
                get_prompt_user_image(previous_context, next_context, html_str, id_str)
            )
    return image_urls, system_prompts, user_prompts


def get_image_summary(image_urls, system_prompts, user_prompts):
    llm = get_gpt()
    multimodal_llm = MultiModal(llm)
    return multimodal_llm.batch(
        image_urls, system_prompts, user_prompts, display_image=False
    )


def add_image_description(answer, _element_content):
    """
    llm 에서 받은 answer를 기반으로 해당 element 찾아 answer 추가
    """
    for html_str in answer:
        soup = BeautifulSoup(html_str, "html.parser")
        id_tag = soup.find("id")  # id 태그 찾기
        if id_tag:
            id_value = id_tag.text.strip()  # 텍스트 추출 후 공백 제거
            # data[id_value]['content_text'] = html_str
            _element_content[int(id_value)]["content_text"] = html_str
            print("id 값:", id_value)
        else:
            print("id 태그를 찾을 수 없습니다.")
    return _element_content


def order_equation_prompt(change_element_content):
    """
    방정식 전후 내용을 포함하여 프롬프트, 내용 정렬
    """
    system_prompts: list[str] = []
    user_prompts: list[str] = []

    for idx, item in enumerate(change_element_content):
        previous_context = (
            change_element_content[idx - 1]["content_text"] if idx - 1 >= 0 else None
        )
        next_context = (
            change_element_content[idx + 1]["content_text"]
            if idx + 1 < len(change_element_content)
            else None
        )
        id_str = item["metadata"]["id"]
        html_str = item["content_text"]
        soup = BeautifulSoup(html_str, "html.parser")
        equation_tag = soup.find(attrs={"data-category": "equation"})
        if equation_tag:
            system_prompts.append(get_prompt_system_equation_summary())
            user_prompts.append(
                get_prompt_user_equation(
                    previous_context, next_context, equation_tag.get_text(), id_str
                )
            )
    return system_prompts, user_prompts


def create_messages(system_prompt, user_prompt):
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_prompt,
                },
            ],
        },
    ]
    return messages


def request_llm_equation(system_prompts, user_prompts):
    """
    llm에게 방정식에대한 설명 요청
    """
    messages = []
    for system_prompt, user_prompt in zip(system_prompts, user_prompts):
        message = create_messages(system_prompt, user_prompt)
        messages.append(message)
    llm = get_gpt()
    return llm.batch(messages)


def add_eqaution_description(answer, change_element_content):
    """
    llm 에서 받은 answer를 기반으로 해당 element 찾아 answer 추가
    """
    for r in answer:
        html_str = r.content
        soup = BeautifulSoup(html_str, "html.parser")
        id_tag = soup.find("id")  # id 태그 찾기
        if id_tag:
            id_value = id_tag.text.strip()  # 텍스트 추출 후 공백 제거
            new_p = soup.new_tag("p")
            new_p.string = html_str
            change_element_content[int(id_value)]["content_text"] = (
                change_element_content[int(id_value)]["content_text"] + html_str
            )
            print("id 값:", id_value)
        else:
            print("id 태그를 찾을 수 없습니다.")
    return change_element_content


def change_json_to_document(final_content):
    docs = []
    for d in final_content:
        doc = Document(page_content=d["content_text"], metadata=d["metadata"])
        docs.append(doc)
    return docs

def get_retriever(split_docs, embedding_path, model="text-embedding-3-small"):
    # PDF 파일 열기
    basename = os.path.splitext(embedding_path)[0]
    embeddings = OpenAIEmbeddings(model=model)
    db = FAISS.from_documents(documents=split_docs, embedding=embeddings)
    db.save_local(basename)
    return db.as_retriever(
        search_type="mmr", search_kwargs={"k": 5, "lambda_mult": 0.25, "fetch_k": 10}
    )


def load_retreiver(embedding_basename, model="text-embedding-3-small"):
    embeddings = OpenAIEmbeddings(model=model)
    db = FAISS.load_local(
        embedding_basename, embeddings, allow_dangerous_deserialization=True
    )
    return db.as_retriever(
        search_type="mmr", search_kwargs={"k": 5, "lambda_mult": 0.25, "fetch_k": 10}
    )


def get_bm25_retriever(split_docs):
    bm25_retriever = BM25Retriever.from_documents(split_docs)
    bm25_retriever.k = 5  # BM25Retriever의 검색 결과 개수를 1로 설정합니다.
    return bm25_retriever


def get_cohere_raranker():
    return cohere.Client(os.getenv("COHERE_API_KEY"))


def get_reranker(esenmble_retriever, user_request):
    docs = esenmble_retriever.invoke(user_request)
    documents_text = [doc.page_content for doc in docs]
    # print(documents_text)
    co = get_cohere_raranker()
    response = co.rerank(
        query=user_request,
        documents=documents_text,
        model="rerank-multilingual-v3.0",
        top_n=6,  # 상위 3개 결과만 리랭킹 반환
    )
    reranked_docs = []
    for result in response.results:
        reranked_docs.append(
            {
                # ** 연산자는 딕셔너리의 키-값 쌍을 풀어헤쳐 새로운 딕셔너리에 넣을 때 사용합니다.
                # 예를 들어, a = {'x':1, 'y':2}, b = {'z':3} 라면 {**a, **b}는 {'x':1, 'y':2, 'z':3}가 됩니다.
                **docs[result.index].dict(),
                "rerank_score": result.relevance_score,
            }
        )
    return reranked_docs


def reorder_documents(docs):
    # 재정렬
    reordering = LongContextReorder()
    reordered_docs = reordering.transform_documents(docs)
    return reordered_docs


def get_esenmble_retriever(retriever1, retriever2):
    ensemble_retriever = EnsembleRetriever(
        retrievers=[retriever1, retriever2],
        weights=[0.6, 0.4],  # 각 리트리버의 가중치를 설정합니다.
        k=6,  # 최종적으로 반환할 문서의 개수를 설정합니다.
    )
    return ensemble_retriever
