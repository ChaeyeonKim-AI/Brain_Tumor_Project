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
                "아래와 같은 형식으로 답변을 작성해주세요. 각 섹션은 반드시 '###'로 구분되어야 합니다:\n"
                "### 분석 결과\n"
                "종양의 위치, 크기, 불규칙성 및 위험도에 대한 분석 결과를 작성해주세요.\n"
                "### 의학적 권고 사항\n"
                "종양의 상태를 고려하여 필요한 의학적 권고 사항을 작성해주세요.\n"
                "### 최종 요약\n"
                "환자에게 전달할 종합적인 요약을 작성해주세요.\n\n"
                "위험도 점수에 따라 다음 카테고리를 참고하세요:\n"
                "- 위험도 점수 <= 33: 낮은 위험도\n"
                "- 34 <= 위험도 점수 <= 66: 중간 위험도\n"
                "- 위험도 점수 > 66: 높은 위험도"
            )
        )

        # LLM 호출 및 응답 처리
        response = chat_model([system_message, user_message])
        response_content = response.content

        # LLM 응답 확인
        st.write("**LLM 응답 확인:**", response_content)

        # 구분자 "###"를 기준으로 파트 나누기
        parts = response_content.split("###")

        # 분석 결과 출력
        st.subheader("🔍 분석 결과")
        if len(parts) > 1:
            st.write(parts[1].strip())
        else:
            st.warning("분석 결과가 존재하지 않습니다.")

        st.divider()

        # 의학적 권고 사항 출력
        st.subheader("🩺 의학적 권고 사항")
        if len(parts) > 2:
            st.write(parts[2].strip())
        else:
            st.warning("의학적 권고 사항이 존재하지 않습니다.")

        st.divider()

        # 최종 요약 출력
        st.subheader("👩‍⚕️ 최종 요약")
        if len(parts) > 3:
            st.write(parts[3].strip())
        else:
            st.warning("최종 요약이 존재하지 않습니다.")
