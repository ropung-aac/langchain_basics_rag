from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import base64
from io import BytesIO
from PIL import Image

load_dotenv()
llm = ChatOpenAI(model="gpt-4o")

st.write("""
# 내분비내과 의사
## 의사 솔루션
""")

# 탭 생성
tab1, tab2 = st.tabs(["고민", "음식 이미지 분석"])

with tab1:
    st.subheader("고민 상담")
    poem = st.text_input("고민을 입력해주세요.")
    
    if st.button("고민 작성하기"):
        with st.spinner("Wait for it...", show_time=True):
            res = llm.invoke(poem + " 다이어트 솔루션을 컨설팅해주세요.")
            st.write(res.content)

with tab2:
    st.subheader("음식 이미지 분석")
    
    # 이미지 업로드
    uploaded_file = st.file_uploader(
        "음식 이미지를 업로드해주세요", 
        type=['png', 'jpg', 'jpeg'],
        help="PNG, JPG, JPEG 파일만 업로드 가능합니다."
    )
    
    if uploaded_file is not None:
        # 이미지 표시
        image = Image.open(uploaded_file)
        st.image(image, caption="업로드된 음식 이미지", use_column_width=True)
        
        # 이미지를 base64로 인코딩
        def encode_image_to_base64(image):
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return img_str
        
        # 음식명 인식 버튼
        if st.button("음식명 인식하기"):
            with st.spinner("이미지를 분석 중입니다...", show_time=True):
                try:
                    # 이미지를 base64로 인코딩
                    base64_image = encode_image_to_base64(image)
                    
                    # HumanMessage로 이미지와 텍스트 프롬프트 함께 보내기
                    message = HumanMessage(
                        content=[
                            {
                                "type": "text",
                                "text": "이 이미지에 있는 음식의 이름을 정확하게 알려주세요. 한국어로 답변해주세요. 여러 음식이 있다면 모두 나열해주세요."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    )
                    
                    # AI에게 이미지 분석 요청
                    response = llm.invoke([message])
                    
                    st.success("음식 인식 완료!")
                    st.write("**인식된 음식:**")
                    st.write(response.content)
                    
                    # 세션 상태에 음식명 저장 (나중에 피드백 생성에 사용)
                    st.session_state.recognized_food = response.content
                    
                except Exception as e:
                    st.error(f"오류가 발생했습니다: {str(e)}")
        
        # 인식된 음식이 있으면 피드백 생성 버튼 표시
        if hasattr(st.session_state, 'recognized_food'):
            st.divider()
            st.write("**인식된 음식에 대한 의사 피드백:**")
            
            if st.button("의사 피드백 받기"):
                with st.spinner("의사 피드백을 생성 중입니다...", show_time=True):
                    try:
                        feedback_prompt = f"""
                        인식된 음식: {st.session_state.recognized_food}
                        
                        위 음식에 대해 내분비내과 의사 관점에서 다음 사항들을 포함하여 피드백을 제공해주세요:
                        1. 영양성분 분석 (칼로리, 탄수화물, 단백질, 지방 등)
                        2. 다이어트에 미치는 영향
                        3. 당뇨병이나 대사질환 환자에게 미치는 영향
                        4. 개선 방안이나 대체 음식 추천
                        5. 섭취 시 주의사항
                        
                        전문적이면서도 이해하기 쉽게 설명해주세요.
                        """
                        
                        feedback_response = llm.invoke(feedback_prompt)
                        st.write(feedback_response.content)
                        
                    except Exception as e:
                        st.error(f"피드백 생성 중 오류가 발생했습니다: {str(e)}")

# 사이드바에 사용법 안내
with st.sidebar:
    st.markdown("""
    ## 사용법
    
    ### 텍스트 고민 탭
    - 다이어트나 건강 관련 고민을 입력하세요
    - AI 의사가 전문적인 조언을 제공합니다
    
    ### 음식 이미지 분석 탭
    1. 음식 사진을 업로드하세요
    2. '음식명 인식하기' 버튼을 클릭하세요
    3. 인식 결과를 확인하세요
    4. '의사 피드백 받기' 버튼으로 전문 조언을 받으세요
    
    ### 지원 파일 형식
    - PNG, JPG, JPEG
    """)