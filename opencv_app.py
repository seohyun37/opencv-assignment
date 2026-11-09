import os
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Streamlit 웹 페이지 기본 설정
st.set_page_config(page_title="ex06 - OpenCV & Streamlit 과제", layout="wide")
st.title("🖼️ ex06: 이미지 처리 및 시각화 애플리케이션")

# 1. 이미지 로드 보조 함수 (한글 경로 및 Streamlit 메모리 파일 지원)
def load_image(img_input):
    if isinstance(img_input, str):
        # 파이썬 경로인 경우
        img_array = np.fromfile(img_input, np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    else:
        # Streamlit 업로드 파일인 경우
        file_bytes = np.asarray(bytearray(img_input.read()), dtype=np.uint8)
        return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

# 2. 사이드바 실습 기능 선택 메뉴
st.sidebar.header("⚙️ 과제 기능 메뉴")
option = st.sidebar.selectbox(
    "실습 항목을 선택하세요",
    [
        "1. 이미지 흑백 (Grayscale) 변환",
        "2. 히스토그램 정규화 & 평활화",
        "3. 이미지 알파 블렌딩 (Alpha Blending)"
    ]
)

# -------------------------------------------------------------
# [기능 1] 이미지 흑백 변환
# -------------------------------------------------------------
if option == "1. 이미지 흑백 (Grayscale) 변환":
    st.subheader("1. 이미지 그레이스케일 변환")
    uploaded_file = st.file_uploader("이미지 파일을 선택하세요", type=["jpg", "jpeg", "png", "bmp", "webp"])
    
    if uploaded_file is not None:
        img = load_image(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="원본 컬러 이미지", use_column_width=True)
        with col2:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            st.image(gray_img, caption="그레이스케일 변환 이미지", use_column_width=True)

# -------------------------------------------------------------
# [기능 2] 히스토그램 정규화 & 평활화
# -------------------------------------------------------------
elif option == "2. 히스토그램 정규화 & 평활화":
    st.subheader("2. 히스토그램 정규화(Normalize) 및 평활화(Equalize)")
    uploaded_file = st.file_uploader("이미지 파일을 선택하세요", type=["jpg", "jpeg", "png", "bmp", "webp"])
    
    if uploaded_file is not None:
        # 그레이스케일로 로드
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        # 정규화 및 평활화 연산
        img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img_eq = cv2.equalizeHist(img)
        
        # 이미지 결과 비교
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(img, caption="Before (원본)", use_column_width=True)
        with c2:
            st.image(img_norm, caption="Normalize (정규화)", use_column_width=True)
        with c3:
            st.image(img_eq, caption="Equalize (평활화)", use_column_width=True)
            
        # 히스토그램 분포 그래프 시각화
        st.write("### 📊 히스토그램 분포 비교")
        fig, ax = plt.subplots(1, 3, figsize=(12, 3))
        ax[0].plot(cv2.calcHist([img], [0], None, [256], [0, 256]), color='black')
        ax[0].set_title("Before")
        ax[1].plot(cv2.calcHist([img_norm], [0], None, [256], [0, 256]), color='blue')
        ax[1].set_title("Normalize")
        ax[2].plot(cv2.calcHist([img_eq], [0], None, [256], [0, 256]), color='red')
        ax[2].set_title("Equalize")
        st.pyplot(fig)

# -------------------------------------------------------------
# [기능 3] 이미지 알파 블렌딩
# -------------------------------------------------------------
elif option == "3. 이미지 알파 블렌딩 (Alpha Blending)":
    st.subheader("3. 두 이미지의 가중치 합성 (Alpha Blending)")
    
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        file1 = st.file_uploader("첫 번째 이미지", type=["jpg", "png", "jpeg", "webp"], key="img1")
    with col_u2:
        file2 = st.file_uploader("두 번째 이미지", type=["jpg", "png", "jpeg", "webp"], key="img2")
    
    if file1 and file2:
        img1 = load_image(file1)
        img2 = load_image(file2)
        
        # 알파 블렌딩 연산을 위해 두 이미지의 해상도 통일
        h, w, c = img1.shape
        img2 = cv2.resize(img2, (w, h))
        
        # 가중치(Alpha) 조절 슬라이더
        alpha = st.slider("투명도 비율 (Alpha)", 0.0, 1.0, 0.5, step=0.02)
        
        # 가중치 합성 연산
        dst = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        st.image(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB), caption=f"합성 결과 (Alpha = {alpha:.2f})", use_column_width=True)