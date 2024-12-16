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
                "아래와 같은 형식으로 답변을 작성해주세요. 각 섹션은 반드시 '###'로 구분되어야 하며, "
                "내용은 환자 데이터를 기반으로 작성하고 의학적 권고 사항은 필요에 따라 여러 항목으로 나누어 넘버링하세요:\n\n"
                "###\n"
                "1. 종양 중심과 뇌 중심 간 거리:\n"
                "   - 평균 거리 값은 약 20.74 픽셀이며, 최소 4.08 픽셀에서 최대 58.55 픽셀로 관찰됩니다.\n"
                "   - 종양 중심이 뇌 중심과의 거리가 **{distance:.2f} 픽셀**인 경우:\n"
                "       - 이 값이 평균(20.74 픽셀)보다 **높다면**, 뇌 중심으로부터 **상대적으로 멀리 위치**해 있는 것으로 볼 수 있습니다.\n"
                "       - 특히, Q3 (23.31 픽셀)를 초과하는 경우 뇌 중심에서 **더 먼 위치**로 평가됩니다.\n"
                "       - 최댓값(58.55 픽셀)에 근접하면 뇌의 외곽에 위치할 가능성이 높습니다.\n"
                "   - 반대로, 거리가 **Q1 이하(15.79 픽셀)**라면 종양이 뇌의 중심부에 가까워 중요한 뇌 영역을 침범할 위험이 높습니다.\n\n"
                "   - 평균 거리 값은 약 20.74 픽셀이며, 최소 4.08 픽셀에서 최대 58.55 픽셀로 관찰됩니다.\n"
                "   - 종양 중심이 뇌 중심과의 거리가 **{distance:.2f} 픽셀**인 경우, 해당 값이 평균과 가까우면 일반적인 위치를 보입니다.\n"
                "   - 거리가 **{distance:.2f} 픽셀**로 1사분위 (15.79) 이하로 낮다면 뇌의 중심부에 가까워 중요한 뇌 영역을 침범할 위험이 있습니다.\n\n"
                "2. 종양 크기:\n"
                "   - 평균 크기는 **455.9 픽셀**이며, 최소 1 픽셀에서 최대 1524 픽셀까지 다양합니다.\n"
                "   - 현재 종양 크기가 **{size} 픽셀**인 경우, 일반적인 크기 (Q1: 161.75 ~ Q3: 670) 내에 속하는지 확인하세요.\n"
                "   - 만약 종양 크기가 Q3 (670) 이상이면 빠른 성장이 의심되며, 악성 가능성을 평가해야 합니다.\n\n"
                "3. 종양 불규칙성:\n"
                "   - 평균 꼭짓점 수는 **9.2개**이며, 최소 1개에서 최대 15개까지 분포합니다.\n"
                "   - 종양의 불규칙성이 **{irregularity}**로 평균보다 높거나 Q3 (11개) 이상일 경우 악성 가능성이 있습니다.\n"
                "   - 경계 비대칭이 심하면 종양의 유형에 대한 정밀 진단이 필요합니다.\n\n"
                "4. 위험도 점수:\n"
                "   - 위험도 점수의 평균은 **28.97**이며, Q3는 41.5로 나타납니다.\n"
                "   - 환자의 위험도 점수가 **{risk_score:.2f}**일 경우, 다음과 같은 기준에 따라 평가하세요:\n"
                "      - **1 ~ 11.2** (Q1 이하): 낮은 위험도\n"
                "      - **11.3 ~ 24.65** (Q2): 중간 위험도\n"
                "      - **24.66 ~ 41.5** (Q3): 높은 위험도\n"
                "      - **41.6 이상**: 매우 높은 위험도 (즉각적 평가 필요)\n\n"
                "###\n"
                "다음은 종양의 상태를 정확히 평가하고 관리하기 위한 권고 사항입니다.\n"
                "   - 현재 데이터를 기반으로 환자에게 필요한 권고 사항을 작성해주세요. 권고 사항이 여러 개일 경우 "
                "   - 각 항목을 넘버링하여 작성해주세요. 예를 들면:\n"
                "      - 1. **정밀 영상 검사**:\n"
                "      종양의 위치와 침습 상태를 명확히 확인하기 위해 MRI 또는 CT 추가 검사를 권장합니다.\n"
                "      - 2. **종양 조직 검사 (생검)**:\n"
                "      종양의 양성 또는 악성 여부를 확인하기 위해 세포 조직 검사를 권장합니다.\n"
                "      - 3. **전문의 상담**:\n"
                "      신경외과 및 종양 전문의와 협력하여 수술 또는 방사선 치료 계획을 수립해야 합니다.\n\n"
                "###\n"
                "다음은 환자의 뇌종양 상태에 대한 종합적인 요약입니다."
                "   - 종양의 주요 특성, 위험도, 위치 및 필요한 조치를 종합적으로 설명하고, 빠른 진단과 치료의 필요성을 강조해주세요."
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

