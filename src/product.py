# product.py
# terminal command: streamlit run product.py 

import os
import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# 환경 변수 불러오기
openai_api_key = os.getenv("OPENAI_API_KEY")

# LangChain ChatOpenAI 설정
chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, openai_api_key=openai_api_key)

# Streamlit UI

st.title("딥러닝 기반 뇌종양 진단 보고서")

# 사용자 입력값
st.header("MRI 데이터 입력")
distance = st.number_input("**종양 중심과 뇌 중심 간 거리 (픽셀)**", min_value=0.0, format="%.2f")
size = st.number_input("**종양 크기 (픽셀 수)**", min_value=0, step=1)
irregularity = st.number_input("**종양 불규칙성 (꼭짓점 수)**", min_value=0, step=1)
risk_score = st.number_input("**위험도 점수**", min_value=0.0, format="%.2f")

# 버튼 클릭 시 LLM 호출
if st.button("결과 분석"):
    with st.spinner("LLM 모델을 사용하여 분석 중입니다..."):
        # LangChain 메시지 구성
        system_message = SystemMessage(
            content=(
                "당신은 뇌종양 분석을 전문으로 하는 의료 보조원입니다. "
                "MRI 데이터에서 추출한 정보를 바탕으로 환자에게 종양의 위치, 크기, 불규칙성, "
                "그리고 계산된 위험도 점수에 대한 상세한 해석을 제공합니다. "
                "또한, 환자의 상황에 적합한 다음 단계의 권고 사항을 제안해야 합니다."
            )
        )

        user_message = HumanMessage(
            content=(
                f"다음 MRI 데이터에 기반하여 뇌종양에 대한 의학적 해석을 작성해주세요:\n\n"
                f"- 종양 중심과 뇌 중심 간 거리: {distance:.2f} 픽셀\n"
                f"- 종양 크기: {size} 픽셀\n"
                f"- 종양 불규칙성(꼭짓점 개수): {irregularity}\n"
                f"- 위험도 점수: {risk_score:.2f}\n\n"
                "st.subheader("분석 결과") 에 제목이 적혀있으니 '종양의 위치, 크기, 불규칙성 및 위험도에 대한 분석 결과:' 와 같은 제목은 반복해서 적으면 안 되고, 각 파트별 내용만 작성해주세요.\n"
                "아래와 같은 형식으로 답변을 작성해주세요. 각 섹션은 반드시 '###'로 구분되어야 합니다.\n"
                "### 종양의 위치, 크기, 불규칙성 및 위험도에 대한 분석 결과를 작성해주세요.\n"
                "### 종양의 상태를 고려하여 필요한 의학적 권고 사항을 작성해주세요.\n"
                "### 환자에게 전달할 종합적인 요약을 작성해주세요.\n"
            )
        )

        # LLM 호출 및 응답 처리
        response = chat_model([system_message, user_message])
        response_content = response.content

        # 구분자 "###"를 기준으로 파트 나누기
        parts = [part.strip() for part in response_content.split("###") if part.strip()]

        # 파트별 내용 출력 (중복 방지)
        if len(parts) >= 3:
            # 1. 분석 결과
            st.subheader("🔍 분석 결과")
            st.write(parts[0])  # 첫 번째 파트

            st.divider()

            # 2. 의학적 권고 사항
            st.subheader("🩺 의학적 권고 사항")
            st.write(parts[1])  # 두 번째 파트

            st.divider()

            # 3. 최종 요약
            st.subheader("👩‍⚕️ 최종 요약")
            st.write(parts[2])  # 세 번째 파트
        else:
            st.error("LLM 응답이 올바른 형식으로 반환되지 않았습니다. 입력 데이터를 확인해주세요.")

