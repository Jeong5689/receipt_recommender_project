import streamlit as st
from ocr_engine import OCREngine
from recommender import Recommender

st.title("영수증 OCR + 추천 시스템")

ocr_engine = OCREngine()
recommender = Recommender()

uploaded_file = st.file_uploader("영수증 이미지 업로드", type=["jpg","png"])
user_id = st.text_input("User ID 입력", "user_01")

if uploaded_file and user_id:
    items = ocr_engine.extract_items(uploaded_file)
    st.write("추출된 상품:", items)

    rec_items = recommender.recommend(user_id)
    st.write(f"{user_id}님을 위한 추천 상품:", rec_items)