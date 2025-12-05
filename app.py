import json
import re
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# ======================
# 1) 데이터 로드 + 전처리
# ======================

@st.cache_data(show_spinner=False)
def load_data():
    path = "/Users/songhyeonseog/한솔제지_naver_news.json"  # 너의 JSON 경로
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    def clean(t):
        if not isinstance(t, str):
            return ""
        t = re.sub(r"<.*?>", " ", t)
        t = re.sub(r"\s+", " ", t)
        return t.strip()

    df["title_clean"] = df["title"].apply(clean)
    df["desc_clean"] = df["description"].apply(clean)
    df["text"] = (df["title_clean"] + " " + df["desc_clean"]).str.strip()

    vectorizer = TfidfVectorizer(
        max_df=0.8,
        min_df=3,
        ngram_range=(1,2)
    )
    tfidf_matrix = vectorizer.fit_transform(df["text"])

    return df, vectorizer, tfidf_matrix


df, vectorizer, tfidf_matrix = load_data()

# ======================
# 2) 문장 분할
# ======================

def split_sentences(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(?<=[가-힣0-9])[.](?=\s)", ".<eos>", text)
    text = re.sub(r"(?<=[가-힣0-9])다(?=\s)", "다.<eos>", text)
    text = re.sub(r"(?<=[가-힣0-9])요(?=\s)", "요.<eos>", text)
    sents = [s.strip() for s in text.split("<eos>") if len(s.strip()) > 5]
    return sents

# ======================
# 3) 검색 (TF-IDF 기반 RAG)
# ======================

def search_similar(question, top_k=5):
    q_vec = vectorizer.transform([question])
    sims = cosine_similarity(q_vec, tfidf_matrix)[0]
    idx = np.argsort(sims)[::-1][:top_k]

    return [
        {
            "idx": int(i),
            "score": float(sims[i]),
            "text": df.loc[i, "text"],
            "title": df.loc[i, "title_clean"],
            "url": df.loc[i, "link"]
        }
        for i in idx
    ]


# ======================
# 4) RAG 컨텍스트 구성
# ======================

def build_context(question, top_k=5, max_sents=8):
    arts = search_similar(question, top_k=top_k)
    merged = " ".join(a["text"] for a in arts)

    sents = split_sentences(merged)
    if not sents:
        return ""

    sent_vecs = vectorizer.transform(sents)
    q_vec = vectorizer.transform([question])
    sims = cosine_similarity(q_vec, sent_vecs)[0]
    idx = np.argsort(sims)[::-1]

    picked = []
    seen = set()
    for i in idx:
        s = sents[i]
        if s in seen:
            continue
        picked.append(s)
        seen.add(s)
        if len(picked) >= max_sents:
            break

    return "\n".join(picked)


# ======================
# 5) HF LLM 연결
# ======================

HF_TOKEN = "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"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)

def call_llm(prompt):
    resp = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": "너는 한솔 그룹 분석 전문 데이터 컨설턴트야."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=300,
        temperature=0.2,
    )
    return resp.choices[0].message.content


# ======================
# 6) 최종 RAG 답변
# ======================

def rag_answer(question):
    ctx = build_context(question)

    if not ctx:
        return "관련 정보를 찾을 수 없습니다."

    prompt = f"""
아래는 한솔 관련 최근 1000개 뉴스에서 추출한 핵심 문장들이야.
이 문장을 근거로, 질문에 3~5문장으로 명확히 답변해줘.

[질문]
{question}

[컨텍스트]
{ctx}
"""
    return call_llm(prompt)


# ======================
# 7) Streamlit UI
# ======================

st.set_page_config(page_title="한솔 뉴스 RAG 챗봇", layout="wide")
st.title("📊 한솔 소식 분석 RAG 챗봇")
st.write("네이버 뉴스 1000건 기반·TF-IDF·LLM 결합 모델")

user_input = st.chat_input("궁금한 질문을 입력하세요. 예: '한솔제지 요즘 문제는?'")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("생각 중..."):
            answer = rag_answer(user_input)
            st.write(answer)
