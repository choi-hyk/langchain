import os
from dotenv import load_dotenv
import feedparser
from typing import List, Optional
from html import unescape
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

RSS_URL = "https://v2.velog.io/rss/@choi-hyk"

# lxml이 없으면 html.parser로 자동 폴백
try:
    import lxml  # noqa

    PARSER = "lxml"
except Exception:
    PARSER = "html.parser"


def pick_html_from_entry(e) -> Optional[str]:
    # 1순위 content:encoded
    if getattr(e, "content", None):
        contents = e.content if isinstance(e.content, list) else [e.content]
        # text/html 타입 우선
        html_pref = [
            c.value
            for c in contents
            if getattr(c, "type", "") in ("text/html", "application/xhtml+xml")
        ]
        if html_pref:
            return html_pref[0]
        return contents[0].value

    # 2순위 description
    if getattr(e, "description", None):
        return e.description

    # 3순위 summary
    if getattr(e, "summary", None):
        return e.summary

    return None


def html_to_text(html: Optional[str]) -> str:
    if not html:
        return ""
    # HTML 엔티티 복원 후 파싱
    soup = BeautifulSoup(unescape(html), PARSER)
    # 조각에는 클래스가 없을 수 있으므로 통으로 텍스트화
    text = soup.get_text("\n", strip=True)
    # 자주 섞이는 제로폭과 nbsp 제거
    return text.replace("\u200b", " ").replace("\xa0", " ").strip()


def build_docs_from_rss(rss_url: str) -> List[Document]:
    feed = feedparser.parse(rss_url)
    docs: List[Document] = []
    for e in feed.entries:
        html = pick_html_from_entry(e)
        text = html_to_text(html)
        link = getattr(e, "link", None)
        if text:
            docs.append(
                Document(page_content=text, metadata={"source": link, "via": "rss"})
            )
    return docs


docs = build_docs_from_rss(RSS_URL)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", " ", ""],
)
splits = splitter.split_documents(docs)

emb = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vectordb = FAISS.from_documents(splits, emb)
retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.5},
)

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# 1) LLM 초기화
model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

# 2) 프롬프트
SYSTEM = """당신은 사용자의 Velog 글만 근거로 답한다
지침
- 추정 금지 컨텍스트 외 사실 언급 금지
- 한국어 간결 답변 약어 첫 등장에 풀네임 병기
- 마지막에 '출처' 목록을 unique 링크로 표기
- 확실하지 않으면 모른다고 말한다
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM),
        (
            "human",
            "질문\n{input}\n\n컨텍스트\n{context}\n\n요구 형식\n- 핵심 답변 5줄 이내\n- 빈 줄\n- 출처: 각 줄에 하나의 링크",
        ),
    ]
)

# 3) 문서 결합 체인과 RAG 체인
doc_chain = create_stuff_documents_chain(model, prompt)
rag_chain = create_retrieval_chain(retriever, doc_chain)

query = "글쓴이는 디자인패턴 정리를 얼마나 진행했어?"
out = rag_chain.invoke({"input": query})

print("\n=== 답변 ===\n")
print(out["answer"])
