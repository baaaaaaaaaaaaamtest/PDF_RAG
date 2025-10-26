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


# def split_pdf(filepath, batch_size=5):
#     """
#     입력 PDF를 분할 PDF 파일로 분할
#     """
#     # PDF 파일 열기
#     output_file_basename = os.path.splitext(filepath)[0]
#     os.makedirs(output_file_basename, exist_ok=True)
#     input_pdf = pymupdf.open(filepath)
#     num_pages = len(input_pdf)
#     print(f"총 페이지 수: {num_pages}")

#     ret = []
#     # PDF 분할
#     for start_page in range(0, num_pages, batch_size):
#         end_page = min(start_page + batch_size, num_pages) - 1
#         # 분할된 PDF 저장
#         # /folder/example.pdf = > ['/folder/example','.pdf']
#         output_file = f"{output_file_basename}_{start_page:04d}_{end_page:04d}.pdf"
#         print(f"분할 PDF 생성: {output_file}")
#         with pymupdf.open() as output_pdf:
#             output_pdf.insert_pdf(input_pdf, from_page=start_page, to_page=end_page)
#             output_pdf.save(output_file)
#             ret.append(output_file)

#     # 입력 PDF 파일 닫기
#     input_pdf.close()

#     return ret, output_file_basename


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


# def reOrder_id(analysis_file_path):
#     element_content = []
#     page_range = 5
#     last_number = 0
#     for i, path in enumerate(sorted(analysis_file_path)):
#         with open(path, "r", encoding="utf-8") as file:
#             # 파일 내용을 파이썬 객체로 변환
#             data = json.load(file)  # 보통 JSON 배열이면 list 타입이 됨
#             change_page = page_range * i
#             print(last_number)
#             for j, element in enumerate(data["elements"]):
#                 element["id"] = last_number + j
#                 html_content = element["content"]["html"]
#                 soup = BeautifulSoup(html_content, "html.parser")
#                 tag = soup.find(attrs={"id": True})
#                 tag["id"] = last_number + j
#                 element["content"]["html"] = str(tag)
#                 element["page"] = change_page + element["page"]
#             last_number = len(data["elements"])
#             element_content.extend(data["elements"])
#     return element_content

# class LayoutAnalyzer:
#     analyzed_files = []

#     def __init__(self, api_key):
#         self.api_key = api_key

#     def _upstage_layout_analysis(self, input_file):
#         """
#         레이아웃 분석 API 호출

#         :param input_file: 분석할 PDF 파일 경로
#         :param output_file: 분석 결과를 저장할 JSON 파일 경로
#         """
#         # API 요청 보내기
#         response = requests.post(
#             # "https://api.upstage.ai/v1/document-ai/layout-analysis",
#             "https://api.upstage.ai/v1/document-ai/document-parse",
#             headers={"Authorization": f"Bearer {os.environ.get('UPSTAGE_API_KEY')}"},
#             data={"ocr": False},
#             files={"document": open(input_file, "rb")},
#         )

#         # 응답 저장
#         if response.status_code == 200:
#             output_file = os.path.splitext(input_file)[0] + ".json"
#             with open(output_file, "w", encoding="utf-8") as f:
#                 json.dump(response.json(), f, ensure_ascii=False)
#             return output_file
#         else:
#             raise ValueError(f"예상치 못한 상태 코드: {response.status_code}")

#     def execute(self, split_files):
#         for file in split_files:
#             self.analyzed_files.append(self._upstage_layout_analysis(file))
#         return self.analyzed_files


class PDFImageProcessor:
    """
    PDF 이미지 처리를 위한 클래스

    PDF 파일에서 이미지를 추출하고, HTML 및 Markdown 형식으로 변환하는 기능을 제공합니다.
    """

    def __init__(self, pdf_file):
        """
        PDFImageProcessor 클래스의 생성자

        :param pdf_file: 처리할 PDF 파일의 경로
        """
        self.pdf_file = pdf_file
        self.json_files = sorted(glob(os.path.splitext(pdf_file)[0] + "*.json"))
        self.output_folder = os.path.splitext(pdf_file)[0]
        self.filename = os.path.splitext(os.path.basename(pdf_file))[0]

    @staticmethod
    def _load_json(json_file):
        """
        JSON 파일을 로드하는 정적 메서드

        :param json_file: 로드할 JSON 파일의 경로
        :return: JSON 데이터를 파이썬 객체로 변환한 결과
        """
        with open(json_file, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _get_page_sizes(json_data):
        """
        각 페이지의 크기 정보를 추출하는 정적 메서드

        :param json_data: JSON 데이터
        :return: 페이지 번호를 키로, [너비, 높이]를 값으로 하는 딕셔너리
        """
        page_sizes = {}
        for page_element in json_data["metadata"]["pages"]:
            width = page_element["width"]
            height = page_element["height"]
            page_num = page_element["page"]
            page_sizes[page_num] = [width, height]
        return page_sizes

    def pdf_to_image(self, page_num, dpi=300):
        """
        PDF 파일의 특정 페이지를 이미지로 변환하는 메서드

        :param page_num: 변환할 페이지 번호 (1부터 시작)
        :param dpi: 이미지 해상도 (기본값: 300)
        :return: 변환된 이미지 객체
        """
        with pymupdf.open(self.pdf_file) as doc:
            # print(doc)
            page = doc[page_num - 1].get_pixmap(dpi=dpi)
            target_page_size = [page.width, page.height]
            page_img = Image.frombytes("RGB", target_page_size, page.samples)
            # page_img.save(f'{page_num}' + ".png")  # 이미지 저장
        return page_img

    @staticmethod
    def normalize_coordinates(coordinates, output_page_size):
        """
        좌표를 정규화하는 정적 메서드

        :param coordinates: 원본 좌표 리스트
        :param output_page_size: 출력 페이지 크기 [너비, 높이]
        :return: 정규화된 좌표 (x1, y1, x2, y2)
        """
        x_values = [coord["x"] for coord in coordinates]

        y_values = [coord["y"] for coord in coordinates]
        x1, y1, x2, y2 = min(x_values), min(y_values), max(x_values), max(y_values)

        return (
            x1 / output_page_size[0],
            y1 / output_page_size[1],
            x2 / output_page_size[0],
            y2 / output_page_size[1],
        )

    @staticmethod
    def crop_image(img, coordinates, output_file):
        """
        pdf 파일에서 얻은 좌표의 위치를 정규화하여 비율로 변환 한 coordinates
        그 이후 pdf 파일을 이미지화
        이미지의 크기를 사용하여 정규화 값과 곱하여 이미지속 좌표를 재생산하여 크롭하는 로직


        :param img: 원본 이미지 객체
        :param coordinates: 정규화된 좌표 (x1, y1, x2, y2)
        :param output_file: 저장할 파일 경로
        """
        img_width, img_height = img.size
        x1, y1, x2, y2 = [
            int(coord * dim)
            for coord, dim in zip(coordinates, [img_width, img_height] * 2)
        ]
        cropped_img = img.crop((x1, y1, x2, y2))
        cropped_img.save(output_file)

    def change_image_path(self, element, output_file, output_folder):
        """
        upstage 에서 받아온 json 파일의 target의 img 태그 src 를 sample-report 폴더 경로로 변경하는 코드

        Args:
        element : upstage에서 받아온 내부에 저장된 json
        output_file : 파일명
        output_folder : 파일 경로

        Return : 변경된 파일 경로
        """
        soup = BeautifulSoup(element["html"], "html.parser")
        img_tag = soup.find("img")
        # print(img_tag)
        if img_tag:
            # 상대 경로로 변경
            relative_path = os.path.relpath(output_file, output_folder)
            img_tag["src"] = relative_path.replace("\\", "/")
        return str(soup)

    def change_add_id(self, data_list):
        html_content = []
        for i, data in enumerate(data_list):
            soup = BeautifulSoup(data, "html.parser")

            for tag in soup.find_all(attrs={"id": True}):
                tag["id"] = str(i)
                html_content.append(str(tag))
        return html_content

    def extract_images(self):
        """
        전체 이미지 처리 과정을 실행하는 메서드

        PDF에서 이미지를 추출하고, HTML 및 Markdown 파일을 생성합니다.
        """
        figure_count = {}  # 페이지별 figure 카운트를 저장하는 딕셔너리

        output_folder = self.output_folder

        # return output_folder
        os.makedirs(output_folder, exist_ok=True)

        print(f"폴더가 생성되었습니다: {output_folder}")
        html_content = []  # HTML 내용을 저장할 리스트
        for json_file in self.json_files:
            json_data = self._load_json(json_file)
            page_sizes = self._get_page_sizes(json_data)
            # print(f"page_sizes : {page_sizes}")
            # 파일 이름에서 페이지 범위 추출
            page_range = os.path.basename(json_file).split("_")[1:]
            # print(f"page_range : {page_range}")
            start_page = int(page_range[0])
            # print(f"start_page : {start_page}")
            for element in json_data["elements"]:
                if element["category"] == "chart":
                    # 파일 내에서의 상대적인 페이지 번호 계산
                    relative_page = element["page"]
                    # print(f"relative_page : {relative_page}")
                    page_num = start_page + relative_page
                    # print(f"page_num : {page_num}")
                    coordinates = element["bounding_box"]
                    # print(f"coordinates : {coordinates}")
                    output_page_size = page_sizes[relative_page]
                    # print(f"output_page_size : {output_page_size}")
                    pdf_image = self.pdf_to_image(page_num)
                    # print(pdf_image)
                    normalized_coordinates = self.normalize_coordinates(
                        coordinates, output_page_size
                    )
                    # print(f"normalized_coordinates : {normalized_coordinates}")
                    # 페이지별 figure 카운트 관리
                    if page_num not in figure_count:
                        figure_count[page_num] = 1
                    else:
                        figure_count[page_num] += 1

                    # 출력 파일명 생성
                    output_file = os.path.join(
                        output_folder,
                        f"page_{page_num}_chart_{figure_count[page_num]}.png",
                    )
                    # print(f"이미지 저장됨: {output_file}")

                    self.crop_image(pdf_image, normalized_coordinates, output_file)
                    # HTML에서 이미지 경로 업데이트
                    element["html"] = self.change_image_path(
                        element, output_file, output_folder
                    )
                    # print(element["html"])
                html_content.append(element["html"])
        html_content = self.change_add_id(html_content)
        # print(html_content)
        # # HTML 파일 저장
        html_output_file = os.path.join(output_folder, f"{self.filename}.html")

        # print(f'html_content : {html_content[0]}')
        # print(f'html_content : {html_content[15]}')
        # print(f'html_content : {len(html_content)}')

        combined_html_content = "\n".join(
            [item for item in html_content if item is not None]
        )
        soup = BeautifulSoup(combined_html_content, "html.parser")
        all_tags = set([tag.name for tag in soup.find_all()])
        html_tag_list = [tag for tag in list(all_tags) if tag not in ["br"]]

        with open(html_output_file, "w", encoding="utf-8") as f:
            f.write(combined_html_content)

        print(f"HTML 파일이 {html_output_file}에 저장되었습니다.")

        # # Markdown 파일 저장
        md_output_file = os.path.join(output_folder, f"{self.filename}.md")

        md_output = markdown(
            combined_html_content,
            convert=html_tag_list,
        )

        with open(md_output_file, "w", encoding="utf-8") as f:
            f.write(md_output)

        print(f"Markdown 파일이 {md_output_file}에 저장되었습니다.")
        return combined_html_content


def get_summary_from_ai(image_paths, previous_context, next_context):
    llm = get_gpt()
    system_prompt = """You are an expert in extracting useful information from IMAGE.
    With a given image, your task is to extract key entities, summarize them, and write useful information that can be used later for retrieval."""
    user_prompt_template = f"""Here is the context related to the image: 
    previous text : {previous_context}
    
    next text : {next_context}
    
    ###
    Output Format:
    <image>
    <title>
    <summary>
    <entities> 
    <path> {image_paths} </path>
    </image>
    """
    multimodal_llm = MultiModal(llm)
    answer = multimodal_llm.invoke(
        image_paths, system_prompt, user_prompt_template, display_image=False
    )
    return answer


def add_summary_to_html(html_content, output_folder):
    soup = BeautifulSoup(html_content, "html.parser")
    tags = sorted(soup.find_all(attrs={"id": True}), key=lambda x: int(x["id"]))
    for tag in tags:
        if tag.get("data-category") == "chart":
            img_tag = tag.find("img")
            if img_tag and img_tag.has_attr("src"):
                cur_id = int(tag.get("id"))
                image_path = img_tag["src"]
                pre_tag = soup.find(id=cur_id - 1)
                next_tag = soup.find(id=cur_id + 1)
                summary = get_summary_from_ai(
                    f"{output_folder}/{image_path}",
                    pre_tag.get_text(),
                    next_tag.get_text(),
                )
                # print(summary)

                # img 태그는 유지하면서 요약 텍스트를 새로운 <p> 태그로 추가
                summary_tag = soup.new_tag("p")
                summary_tag.string = summary
                img_tag.insert_after(summary_tag)

    with open(f"{output_folder}/analysis.html", "w", encoding="utf-8") as f:
        f.write(str(soup))


def get_docs(output_folder):
    with open(f"{output_folder}/analysis.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    headers_to_split_on = [
        ("h1", "Header 1"),  # 분할할 헤더 태그와 해당 헤더의 이름을 지정합니다.
        ("h2", "Header 2"),
        ("h3", "Header 3"),
    ]
    # 지정된 헤더를 기준으로 HTML 텍스트를 분할하는 HTMLHeaderTextSplitter 객체를 생성합니다.
    html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    # HTML 문자열을 분할하여 결과를 html_header_splits 변수에 저장합니다.
    html_header_splits = html_splitter.split_text(html_content)
    # 분할된 결과를 출력합니다.
    last_segment = os.path.basename(output_folder) + "analysis.html"
    documents = []
    for chunk in html_header_splits:
        doc = Document(
            page_content=chunk.page_content, metadata={"source": last_segment}
        )
        documents.append(doc)
    return documents


def get_retriever(split_docs, model="text-embedding-3-small"):
    embeddings = OpenAIEmbeddings(model=model)
    db = FAISS.from_documents(documents=split_docs, embedding=embeddings)
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
