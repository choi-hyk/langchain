from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

long_text = """
대규모 언어 모델(LLM)은 자연어 이해와 생성 능력을 가진 인공지능 모델입니다.
이 모델은 방대한 양의 텍스트 데이터로 사전 학습되며, 다양한 언어 관련 작업에 활용될 수 있습니다.

프롬프트 엔지니어링은 LLM이 원하는 출력을 생성하도록 유도하는 기술입니다.
ReAct는 추론과 행동을 결합하여 복잡한 문제를 해결하는 프레임워크입니다.
이러한 기술들은 LLM의 성능을 극대화하는 데 중요한 역할을 합니다.

RAG(Retrieval-Augmented Generation)는 외부 지식을 활용하여 LLM의 답변을 강화하는 기술입니다.
RAG 시스템은 외부 문서에서 정보를 검색하고, 그 정보를 모델의 컨텍스트에 추가하여 답변의 정확성과 신뢰성을 높입니다.
이 과정에서 텍스트 분할(Text Splitting)은 긴 문서를 작은 청크로 나누어 효율적인 검색을 가능하게 하는 핵심적인 전처리 기술로 사용됩니다.
"""

docs = [Document(page_content=long_text, metadata={"source": "example_document"})]


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=20,
    length_function=len,
)

chunks = text_splitter.split_documents(docs)

print(f"청크 개수: {len(chunks)}\n")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding_model)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

openai_model = init_chat_model("gpt-4o-mini", model_provider="openai")

qa_chain = RetrievalQA.from_chain_type(
    llm=openai_model,
    chain_type="stuff",
    retriever=retriever,
)

query = "RAG가 무엇인지 설명해줘."

print(f"질문: {query}")
response = qa_chain.invoke(query)
print(f"답변: {response}")

print("\n--- 검색된 청크(문맥) ---")
retrieved_docs = retriever.invoke(query)
for i, doc in enumerate(retrieved_docs):
    print(f"청크 {i+1}:\n{doc.page_content}\n")
