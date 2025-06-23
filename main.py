from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document
import base64
from io import BytesIO
from PIL import Image
import os

load_dotenv()

# OpenAI 모델 초기화
llm = ChatOpenAI(model="gpt-4o")
embeddings = OpenAIEmbeddings()

# 우창윤 선생님 스타일 의료 지식 베이스
medical_knowledge = [
    """
    라면과 혈당: 라면은 정제된 밀가루로 만든 면과 나트륨이 많은 스프로 구성되어 있어 혈당을 급격히 올립니다. 
    GI지수가 높아 당뇨병 환자에게는 특히 주의가 필요한 음식입니다. 
    대안으로는 현미면이나 메밀면, 곤약면을 추천합니다.
    """,
    """
    튀김류와 대사: 치킨, 돈가스 등 튀김류는 트랜스지방과 포화지방이 많아 인슐린 저항성을 증가시킵니다.
    체중 증가와 염증 반응을 유발할 수 있어 대사질환 환자에게는 피해야 할 음식입니다.
    대안으로는 구이나 찜 요리를 추천합니다.
    """,
    """
    탄수화물과 혈당: 흰쌀, 흰빵, 떡 등 정제된 탄수화물은 혈당을 빠르게 올립니다.
    현미, 귀리, 통밀빵 등 복합탄수화물로 대체하는 것이 좋습니다.
    식이섬유가 풍부한 채소와 함께 섭취하면 혈당 상승을 완화할 수 있습니다.
    """,
    """
    단맛과 인슐린: 케이크, 과자, 음료수 등 설탕이 많은 음식은 인슐린 분비를 과도하게 자극합니다.
    장기적으로 인슐린 저항성과 제2형 당뇨병 발병 위험을 높입니다.
    과일이나 견과류로 단맛을 대신하는 것을 추천합니다.
    """,
    """
    나트륨과 혈압: 짠 음식은 혈압을 올리고 부종을 유발할 수 있습니다.
    가공식품, 인스턴트 식품에는 숨어있는 나트륨이 많습니다.
    신선한 재료로 집에서 조리하고, 허브나 향신료로 맛을 내는 것이 좋습니다.
    """,
    """
    식이섬유의 중요성: 채소, 과일, 통곡물에 포함된 식이섬유는 혈당 상승을 완화하고 
    포만감을 오래 유지시켜줍니다. 또한 장내 미생물 균형에도 도움이 됩니다.
    하루에 25-30g의 식이섬유 섭취를 권장합니다.
    """,
    """
    단백질과 포만감: 살코기, 생선, 두부, 콩류 등의 단백질은 혈당을 안정시키고
    근육량 유지에 도움이 됩니다. 체중 관리에도 효과적입니다.
    체중 1kg당 0.8-1.2g의 단백질 섭취를 권장합니다.
    """,
    """
    건강한 지방: 올리브오일, 견과류, 아보카도, 등푸른생선의 오메가-3 지방산은
    염증을 줄이고 심혈관 건강에 도움이 됩니다.
    트랜스지방과 과도한 포화지방은 피하는 것이 좋습니다.
    """,
]

@st.cache_resource
def create_vector_store():
    """의료 지식 벡터 저장소 생성"""
    documents = []
    for i, knowledge in enumerate(medical_knowledge):
        doc = Document(
            page_content=knowledge,
            metadata={"source": f"medical_knowledge_{i}"}
        )
        documents.append(doc)
    
    # 텍스트 분할
    text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(documents)
    
    # 벡터 저장소 생성
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

def get_relevant_knowledge(query, vectorstore, k=3):
    """관련 의료 지식 검색"""
    docs = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in docs])

def create_woo_style_prompt(food_name, relevant_knowledge):
    """우창윤 선생님 스타일 프롬프트 생성"""
    return f"""
    당신은 내분비내과 전문의 우창윤 선생님입니다. 다음은 우창윤 선생님의 실제 말투 예시들입니다:

    실제 말투 예시:
    - "안녕하세요, 우창윤입니다. 팥빙수에 연유, 꼭 넣어야 할까요? 한 번쯤 안 넣고 드셔보는 건 어떨까요?"
    - "안녕하세요, 스친님들! 전 내분비내과 전문의로 비만과 대사 질환, 당뇨병에 관심이 많아요."
    - "'당 떨어진다'며 음료나 과자 먹기. 특정약을 복용하는게 아니라면 여러분 괜찮습니다."
    - "그렇치 않답니다 ㅎ 조금만 더 버텨보세요."
    - "사람들의 광기와 센스를 보았다. 훗"

    음식: {food_name}

    관련 의료 지식:
    {relevant_knowledge}

    **중요**: 반드시 우창윤 선생님의 말투로만 답변하세요!

    핵심 말투 요소 (필수 포함):
    - "~죠", "~답니다", "~해보는 건 어떨까요?" 어미 사용
    - "ㅎ", "훗" 감탄사 포함
    - "스친님들", "여러분" 호칭 사용
    - "괜찮습니다", "조금만 더 버텨보세요" 격려 표현
    - 친근하고 부드러운 톤

    답변 형태 예시:
    "안녕하세요, 스친님들! [음식명]을 보니 [첫인상/특징]죠 ㅎ 
    이 음식은 [문제점]라서 [건강에 미치는 영향]답니다~ 
    [구체적 이유]때문에 [주의사항]해요. 
    대신 [대안음식]으로 바꿔보는 건 어떨까요? 
    조금만 더 신경써주시면 분명 좋아질 거예요!"

    라면 예시: "분식집에서 만날 수 있는 혈당 폭탄 음식 중 하나죠 ㅎ 라면은 정제된 밀가루와 나트륨 덩어리라서 혈당을 급격히 올린답니다~ 대신 현미면이나 메밀면으로 바꿔보는 건 어떨까요?"
    """

# Streamlit 앱 시작
st.write("""
# 내분비내과 우창윤 의사 💊
## 친근한 건강 상담 솔루션
""")

# 벡터 저장소 초기화
vectorstore = create_vector_store()

# 탭 생성
tab1, tab2 = st.tabs(["💬 건강 고민", "🍽️ 음식 이미지 분석"])

with tab1:
    st.subheader("건강 고민 상담")
    st.write("다이어트나 건강 관련 고민을 편하게 말씀해주세요 😊")
    
    concern = st.text_area("고민을 입력해주세요:", placeholder="예: 야식을 끊고 싶은데 자꾸 생각나요...")
    
    if st.button("🩺 우창윤 선생님께 상담받기", type="primary"):
        if concern:
            with st.spinner("선생님이 답변을 준비하고 있어요... ⏰"):
                # 관련 지식 검색
                relevant_knowledge = get_relevant_knowledge(concern, vectorstore)
                
                # 우창윤 스타일 프롬프트 생성
                consultation_prompt = f"""
                당신은 내분비내과 전문의 우창윤 선생님입니다. 다음은 우창윤 선생님의 실제 말투 예시들입니다:

                실제 말투 예시:
                - "안녕하세요, 스친님들! 전 내분비내과 전문의로 비만과 대사 질환, 당뇨병에 관심이 많아요."
                - "'당 떨어진다'며 음료나 과자 먹기. 특정약을 복용하는게 아니라면 여러분 괜찮습니다."
                - "그렇치 않답니다 ㅎ 조금만 더 버텨보세요."
                - "사람들의 광기와 센스를 보았다. 훗"

                환자의 고민: {concern}

                관련 의료 지식:
                {relevant_knowledge}

                **중요**: 반드시 우창윤 선생님의 말투로만 답변하세요!

                핵심 말투 요소 (필수 포함):
                - "안녕하세요, 스친님!" 같은 친근한 인사로 시작
                - "~죠", "~답니다", "~해보는 건 어�을까요?" 어미 사용
                - "ㅎ", "훗" 감탄사 자주 포함
                - "괜찮습니다", "조금만 더 버텨보세요" 격려 표현
                - "그렇치 않답니다", "맞답니다" 등의 표현
                - 친근하고 부드러운 톤 유지

                상담 형태 예시:
                "안녕하세요, 스친님! [공감 표현]죠 ㅎ 
                [고민에 대한 이해] 많은 분들이 그렇답니다~ 
                [전문적 설명과 조언]해요. 
                [구체적 해결방법] 해보는 건 어떨까요? 
                괜찮습니다, 조금만 더 버텨보시면 분명 좋아질 거예요!"

                반드시 우창윤 선생님의 따뜻하고 친근한 말투로 상담해주세요.
                """
                
                response = llm.invoke(consultation_prompt)
                
                st.success("상담 완료! 📋")
                st.write("**💬 우창윤 선생님의 따뜻한 상담:**")
                
                # 답변을 예쁘게 표시
                with st.container():
                    st.markdown("---")
                    # 우창윤 선생님 스타일 박스
                    st.markdown(f"""
                    <div style='background-color: #f0f8ff; padding: 20px; border-radius: 10px; border-left: 4px solid #4CAF50;'>
                        <p style='margin: 0; line-height: 1.6;'>{response.content}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("---")
        else:
            st.warning("고민을 입력해주세요!")

with tab2:
    st.subheader("음식 이미지 분석 🔍")
    st.write("음식 사진을 올려주시면 우창윤 선생님이 친근하게 분석해드려요!")
    
    # 이미지 업로드
    uploaded_file = st.file_uploader(
        "🍽️ 음식 이미지를 업로드해주세요", 
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
        
        # 우창윤 선생님 피드백 받기 버튼
        if st.button("🩺 우창윤 선생님 피드백 받기", type="primary", use_container_width=True):
            with st.spinner("우창윤 선생님이 이미지를 보고 분석하고 계세요... 👨‍⚕️"):
                try:
                    # 이미지를 base64로 인코딩
                    base64_image = encode_image_to_base64(image)
                    
                    # 우창윤 선생님 스타일 이미지 분석 프롬프트
                    image_analysis_message = HumanMessage(
                        content=[
                            {
                                "type": "text",
                                "text": """
                                당신은 내분비내과 전문의 우창윤 선생님입니다. 다음은 우창윤 선생님의 실제 말투 예시들입니다:

                                실제 말투 예시:
                                - "안녕하세요, 스친님들! 전 내분비내과 전문의로 비만과 대사 질환, 당뇨병에 관심이 많아요."
                                - "팥빙수에 연유, 꼭 넣어야 할까요? 한 번쯤 안 넣고 드셔보는 건 어떨까요?"
                                - "'당 떨어진다'며 음료나 과자 먹기. 특정약을 복용하는게 아니라면 여러분 괜찮습니다."
                                - "그렇치 않답니다 ㅎ 조금만 더 버텨보세요."
                                - "사람들의 광기와 센스를 보았다. 훗"

                                이 음식 이미지를 보고 우창윤 선생님의 친근한 말투로 분석해주세요.

                                **중요**: 반드시 우창윤 선생님의 말투로만 답변하세요!

                                핵심 말투 요소 (필수 포함):
                                - "안녕하세요, 스친님들!" 같은 친근한 인사로 시작
                                - "~죠", "~답니다", "~해보는 건 어떨까요?" 어미 사용
                                - "ㅎ", "훗" 감탄사 자주 포함
                                - "괜찮습니다", "조금만 더 버텨보세요" 격려 표현
                                - "그렇치 않답니다", "맞답니다" 등의 표현
                                - 친근하고 부드러운 톤 유지

                                답변 형태:
                                "안녕하세요, 스친님들! [음식명]을/를 보니 [첫인상/특징]죠 ㅎ 
                                이 음식은 [영양학적 분석]라서 [건강에 미치는 영향]답니다~ 
                                [구체적 이유]때문에 [주의사항]해요. 
                                대신 [대안음식/개선방법]으로 바꿔보는 건 어떨까요? 
                                괜찮습니다, 조금만 더 신경써주시면 분명 좋아질 거예요!"

                                라면 예시: "안녕하세요, 스친님들! 라면을 보니 분식집에서 만날 수 있는 혈당 폭탄 음식 중 하나죠 ㅎ 정제된 밀가루와 나트륨 덩어리라서 혈당을 급격히 올린답니다~ 대신 현미면이나 메밀면으로 바꿔보는 건 어떨까요?"

                                반드시 우창윤 선생님의 따뜻하고 친근한 말투로 음식을 분석해주세요.
                                """
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    )
                    
                    # AI에게 우창윤 선생님 스타일 분석 요청
                    feedback_response = llm.invoke([image_analysis_message])
                    
                    # 최종 결과 표시
                    st.success("분석 완료! 🎯")
                    st.write("**💬 우창윤 선생님의 친근한 피드백:**")
                    
                    # 답변을 예쁘게 표시
                    with st.container():
                        st.markdown("---")
                        # 우창윤 선생님 스타일 박스
                        st.markdown(f"""
                        <div style='background-color: #f0f8ff; padding: 20px; border-radius: 10px; border-left: 4px solid #4CAF50; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                            <div style='display: flex; align-items: center; margin-bottom: 10px;'>
                                <span style='font-size: 20px; margin-right: 8px;'>👨‍⚕️</span>
                                <strong style='color: #2E7D32;'>우창윤 선생님</strong>
                            </div>
                            <p style='margin: 0; line-height: 1.6; color: #333;'>{feedback_response.content}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown("---")
                    
                except Exception as e:
                    st.error(f"분석 중 오류가 발생했습니다: {str(e)}")
                    st.write("다시 시도해보시거나 다른 이미지를 업로드해주세요.")

# 사이드바에 사용법 안내
with st.sidebar:
    st.markdown("""
    ## 👨‍⚕️ 우창윤 선생님 소개
    
    안녕하세요! 내분비내과 전문의로 
    비만과 대사 질환, 당뇨병에 관심이 많아요 😊
    
    ## 📖 사용법
    
    ### 💬 건강 고민 탭
    - 다이어트나 건강 관련 고민을 자유롭게 적어주세요
    - AI가 우창윤 선생님 스타일로 따뜻한 조언을 드려요
    
    ### 🍽️ 음식 이미지 분석 탭
    1. 📸 음식 사진을 업로드하세요
    2. 🩺 '우창윤 선생님 피드백 받기' 버튼을 클릭하세요
    3. 친근한 우창윤 선생님의 분석을 받아보세요!
    
    ### 📁 지원 파일 형식
    - PNG, JPG, JPEG
    
    ---
    
    💡 **팁**: 음식 사진은 가능한 한 음식이 잘 보이도록 
    촬영해주시면 더 정확한 분석이 가능해요!
    
    🎯 **간편함**: 이제 사진 업로드 후 버튼 한 번으로 
    우창윤 선생님의 친근한 피드백을 바로 받아보세요!
    
    ---
    
    ⚠️ **주의사항**: 
    이 서비스는 일반적인 건강 정보 제공을 목적으로 하며, 
    실제 진료나 처방을 대체할 수 없습니다.
    """)
    
    # 지식 베이스 정보
    with st.expander("🧠 AI 지식 베이스"):
        st.write(f"현재 {len(medical_knowledge)}개의 의료 지식이 저장되어 있어요")
        st.write("- 혈당과 음식의 관계")
        st.write("- 대사질환과 식단")
        st.write("- 영양소별 건강 효과")
        st.write("- 건강한 대안 음식들")

# 메인 페이지 하단에 추가 정보
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>💙 건강한 식단으로 더 나은 내일을 만들어가요!</p>
    <p>궁금한 점이 있으시면 언제든 상담받아보세요 😊</p>
</div>
""", unsafe_allow_html=True)