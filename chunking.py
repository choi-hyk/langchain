from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    HTMLHeaderTextSplitter,
    MarkdownHeaderTextSplitter,
    Language,
)

text = """
인공지능(Artificial Intelligence, AI)은 인간의 학습 능력, 추론 능력, 지각 능력 등을 모방하는 컴퓨터 과학의 한 분야이다. AI 기술은 1950년대부터 꾸준히 연구되어 왔으며, 최근 딥러닝 기술의 발전으로 인해 비약적인 성장을 이루었다. 특히, 컴퓨터 비전, 자연어 처리, 음성 인식 분야에서 괄목할 만한 성과를 보이고 있다.

자연어 처리(Natural Language Processing, NLP)는 인간의 언어를 컴퓨터가 이해하고 처리하도록 돕는 기술이다. NLP의 핵심적인 발전은 트랜스포머(Transformer) 아키텍처의 등장과 함께 이루어졌다. 이 아키텍처는 문장 내의 단어들 간의 관계를 효과적으로 학습하는 **어텐션(Attention)** 메커니즘을 기반으로 한다.

트랜스포머 아키텍처를 기반으로 한 대규모 언어 모델(Large Language Model, LLM)은 방대한 양의 텍스트 데이터로 사전 학습된 모델이다. GPT-3, PaLM, LLaMA와 같은 모델들이 여기에 속한다. LLM은 문장 완성, 질의응답, 요약, 번역 등 다양한 자연어 처리 작업을 수행할 수 있다.

이러한 LLM을 효과적으로 활용하기 위해 **프롬프트 엔지니어링**이라는 기술이 중요해졌다. 프롬프트 엔지니어링은 모델의 입력(프롬프트)을 조작하여 모델이 원하는 출력을 생성하도록 유도하는 기술이다. 최근에는 **CoT(Chain-of-Thought)**, **ReAct(Reasoning and Acting)**와 같은 고급 프롬프트 기법들이 개발되어 모델의 추론 능력을 극대화하고 있다.

RAG(Retrieval-Augmented Generation)는 외부 지식을 활용하여 LLM의 답변을 강화하는 기술이다. RAG 시스템은 외부 문서에서 정보를 검색하고, 그 정보를 모델의 컨텍스트에 추가하여 답변의 정확성과 신뢰성을 높인다. 이 과정에서 **텍스트 분할(Text Splitting)**은 긴 문서를 작은 청크로 나누어 효율적인 검색을 가능하게 하는 핵심적인 전처리 기술로 사용된다.
"""

print("\n----------------------------------------------\n")

text_splitter = CharacterTextSplitter(
    separator=" ",
    chunk_size=150,
    chunk_overlap=20,
    length_function=len,
    add_start_index=True,
)

docs = text_splitter.create_documents([text])

results = ""

print(f"\n문자 기반으로 분할된 청크의 개수: {len(docs)}\n")
for i, doc in enumerate(docs):
    results += "\n\n" + str(i + 1) + "[ " + doc.page_content + " ] "

print(results)

print("\n----------------------------------------------\n")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=20,
    length_function=len,
    add_start_index=True,
)

docs = text_splitter.create_documents([text])

results = ""

print(f"\n텍스트 구조 기반으로 분할된 청크의 개수: {len(docs)}\n")
for i, doc in enumerate(docs):
    results += "\n\n" + str(i + 1) + "[ " + doc.page_content + " ] "

print(results)

print("\n----------------------------------------------\n")

html_text = """
<!DOCTYPE html>
<html>
<head>
    <title>랭체인 핵심 개념</title>
</head>
<body>
    <h1>랭체인(LangChain)의 개요</h1>
    <p>랭체인은 대규모 언어 모델(LLM) 기반의 애플리케이션을 개발하기 위한 프레임워크입니다.</p>

    <h2>핵심 구성 요소</h2>
    <p>랭체인은 개발자가 복잡한 애플리케이션을 쉽게 만들 수 있도록 돕는 다양한 구성 요소를 제공합니다.</p>
    
    <h3>1. 프롬프트</h3>
    <p>모델에게 지시를 내리는 텍스트입니다. 랭체인에서는 템플릿을 통해 프롬프트를 관리합니다.</p>

    <h3>2. 체인</h3>
    <p>여러 구성 요소를 결합하여 하나의 워크플로우로 만드는 것을 의미합니다.</p>
</body>
</html>
"""

headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
]

html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

docs = html_splitter.split_text(html_text)

print(f"HTML 구조로 분할된 청크의 개수: {len(docs)}")

results = ""

for i, doc in enumerate(docs):
    results += "\n\n" + str(i + 1) + "[ " + doc.page_content + " ] "

print(results)


print("\n----------------------------------------------\n")

markdown_text = """
# 랭체인(LangChain)의 개요

랭체인은 대규모 언어 모델(LLM) 기반의 애플리케이션을 개발하기 위한 프레임워크입니다.

## 핵심 개념
랭체인은 다양한 구성 요소를 제공하여 개발자가 복잡한 애플리케이션을 쉽게 만들 수 있도록 돕습니다.

### 1. 프롬프트
모델에게 지시를 내리는 텍스트입니다. 랭체인에서는 템플릿을 통해 프롬프트를 관리합니다.

### 2. 체인
여러 구성 요소를 결합하여 하나의 워크플로우로 만드는 것을 의미합니다.

## 주요 기능
랭체인은 RAG, 에이전트, 메모리 등 다양한 기능을 지원합니다.
"""

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)


docs = markdown_splitter.split_text(markdown_text)

results = ""

print(f"마크다운 구조로 분할된 청크의 개수: {len(docs)}\n")

for i, doc in enumerate(docs):
    results += "\n\n" + str(i + 1) + "[ " + doc.page_content + " ] "

print(results)

print("\n----------------------------------------------\n")

python_code = """
import numpy as np

def calculate_average(numbers):
    # Calculates the average of a list of numbers
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)

class MyCalculator:
    def __init__(self, value=0):
        self.value = value

    def add(self, num):
        self.value += num
        return self.value

    def subtract(self, num):
        self.value -= num
        return self.value

def main():
    data = [10, 20, 30, 40, 50]
    avg = calculate_average(data)
    print(f"The average is: {avg}")

    calc = MyCalculator(100)
    print(f"Added 20: {calc.add(20)}")
    print(f"Subtracted 50: {calc.subtract(50)}")

if __name__ == "__main__":
    main()
"""

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=200, chunk_overlap=20
)

docs = python_splitter.create_documents([python_code])

results = ""

print(f"파이썬 구조로 분할된 청크의 개수: {len(docs)}\n")
for i, doc in enumerate(docs):
    results += "\n\n" + str(i + 1) + "[ " + doc.page_content + " ] "

print(results)
