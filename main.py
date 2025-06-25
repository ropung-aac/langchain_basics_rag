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

# 우창윤 선생님 스타일 의료 지식 베이스 (RAG용)
medical_knowledge = [
    """라면과 혈당 관리: 라면은 정제된 밀가루로 만든 면과 나트륨이 많은 스프로 구성되어 있어 혈당을 급격히 올립니다. GI지수가 높아 당뇨병 환자에게는 특히 주의가 필요한 음식입니다. 라면 대신 현미면이나 메밀면, 곤약면을 추천합니다. 야채를 많이 넣어 식이섬유를 보충하면 혈당 상승을 완화할 수 있어요.""",
    
    """튀김류와 대사질환: 치킨, 돈가스, 감자튀김 등 튀김류는 트랜스지방과 포화지방이 많아 인슐린 저항성을 증가시킵니다. 체중 증가와 염증 반응을 유발할 수 있어 대사질환 환자에게는 피해야 할 음식입니다. 대안으로는 에어프라이어를 이용한 구이나 찜 요리를 추천합니다. 닭가슴살 구이나 생선구이가 좋은 선택이에요.""",
    
    """정제 탄수화물의 위험성: 흰쌀, 흰빵, 떡, 과자 등 정제된 탄수화물은 혈당을 빠르게 올립니다. 현미, 귀리, 통밀빵, 퀴노아 등 복합탄수화물로 대체하는 것이 좋습니다. 식이섬유가 풍부한 채소와 함께 섭취하면 혈당 상승을 완화할 수 있습니다.""",
    
    """단 음식과 인슐린: 케이크, 과자, 아이스크림, 탄산음료 등 설탕이 많은 음식은 인슐린 분비를 과도하게 자극합니다. 장기적으로 인슐린 저항성과 제2형 당뇨병 발병 위험을 높입니다. 과일이나 견과류, 다크초콜릿으로 단맛을 대신하는 것을 추천합니다.""",
    
    """나트륨과 혈압 관리: 짠 음식은 혈압을 올리고 부종을 유발할 수 있습니다. 가공식품, 인스턴트 식품, 젓갈류에는 숨어있는 나트륨이 많습니다. 신선한 재료로 집에서 조리하고, 허브나 향신료로 맛을 내는 것이 좋습니다.""",
    
    """식이섬유의 중요성: 채소, 과일, 통곡물에 포함된 식이섬유는 혈당 상승을 완화하고 포만감을 오래 유지시켜줍니다. 또한 장내 미생물 균형에도 도움이 됩니다. 하루에 25-30g의 식이섬유 섭취를 권장합니다. 브로콜리, 시금치, 사과, 배 등이 좋은 공급원이에요.""",
    
    """단백질과 근육량 유지: 살코기, 생선, 두부, 콩류 등의 단백질은 혈당을 안정시키고 근육량 유지에 도움이 됩니다. 체중 관리에도 효과적입니다. 체중 1kg당 0.8-1.2g의 단백질 섭취를 권장합니다. 닭가슴살, 연어, 두부, 렌틸콩이 좋은 선택이에요.""",
    
    """건강한 지방 섭취: 올리브오일, 견과류, 아보카도, 등푸른생선의 오메가-3 지방산은 염증을 줄이고 심혈관 건강에 도움이 됩니다. 트랜스지방과 과도한 포화지방은 피하는 것이 좋습니다. 견과류 한 줌, 올리브오일 한 스푼 정도가 적당해요.""",
    
    """패스트푸드의 문제점: 햄버거, 피자, 핫도그 등 패스트푸드는 칼로리, 나트륨, 트랜스지방이 높습니다. 혈당 급상승과 체중 증가를 유발하며 만성질환 위험을 높입니다. 집에서 만든 샐러드나 그릴 요리로 대체하는 것을 추천합니다.""",
    
    """음료와 건강: 탄산음료, 주스, 커피음료에는 많은 설탕이 들어있습니다. 물, 무설탕 차, 탄산수가 가장 좋은 선택입니다. 커피는 하루 2-3잔 이내로 제한하고 설탕 대신 계피나 바닐라 에센스로 맛을 내보세요."""
]

@st.cache_resource
def create_vector_store():
    """의료 지식 벡터 저장소 생성 (RAG 시스템)"""
    documents = []
    for i, knowledge in enumerate(medical_knowledge):
        doc = Document(
            page_content=knowledge,
            metadata={"source": f"medical_knowledge_{i}", "category": "nutrition"}
        )
        documents.append(doc)
    
    # 텍스트 분할 (더 작은 청크로)
    text_splitter = CharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separator="\n"
    )
    splits = text_splitter.split_documents(documents)
    
    # FAISS 벡터 저장소 생성
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

def get_relevant_knowledge(query, vectorstore, k=4):
    """관련 의료 지식 검색 (RAG 검색)"""
    try:
        docs = vectorstore.similarity_search(query, k=k)
        relevant_content = "\n\n".join([doc.page_content for doc in docs])
        return relevant_content
    except Exception as e:
        st.error(f"지식 검색 중 오류: {e}")
        return "일반적인 건강 관리 원칙을 적용하여 답변드리겠습니다."

def create_woo_style_consultation_prompt(concern, relevant_knowledge):
    """우창윤 선생님 스타일 상담 프롬프트 (RAG 적용)"""
    return f"""
    당신은 내분비내과 전문의 우창윤 선생님입니다. 다음은 우창윤 선생님의 실제 말투 예시들입니다:

    실제 말투 예시:
    - "안녕하세요, 위머님 전 내분비내과 전문의로 비만과 대사 질환, 당뇨병에 관심이 많아요."
    - "'당 떨어진다'며 음료나 과자 먹기. 특정약을 복용하는게 아니라면 여러분 괜찮습니다."
    - "그렇지 않답니다 ㅎ 조금만 더 버텨보세요."
    - "사람들의 광기와 센스를 보았다. 훗"
    - "한 번쯤 안 넣고 드셔보는 건 어떨까요?"

    환자의 고민: {concern}

    관련 의료 지식 (RAG 검색 결과):
    {relevant_knowledge}

    **중요**: 반드시 우창윤 선생님의 친근한 말투로만 답변하세요!

    핵심 말투 요소 (필수 포함):
    - "안녕하세요, 위머님!" 같은 친근한 인사로 시작
    - "~죠", "~답니다", "~해보는 건 어떨까요?" 어미 사용
    - 격려 표현
    - "그렇지 않답니다", "맞답니다" 등의 표현
    - 친근하고 부드러운 톤 유지

    상담 형태 예시:
    "안녕하세요, 위머님! [공감 표현]죠 
    [고민에 대한 이해] 많은 분들이 그렇답니다~ 
    [RAG 지식 기반 전문적 설명과 조언]해요. 
    [구체적 해결방법] 해보는 건 어떨까요? 
    괜찮습니다, 조금만 더 버텨보시면 분명 좋아질 거예요!"

    반드시 우창윤 선생님의 따뜻하고 친근한 말투로 상담해주세요.
    """

# Streamlit 앱 시작
st.set_page_config(
    page_title="WIM 닥터프렌즈 우창윤 AI",
    page_icon="👨‍⚕️",
    layout="wide"
)

st.write("""
# 내분비내과 우창윤 의사 👨‍⚕️
## RAG 기반 AI 건강 상담 시스템
""")

# 벡터 저장소 초기화
with st.spinner("의료 지식 데이터베이스를 준비하고 있어요... 🧠"):
    vectorstore = create_vector_store()

st.success("RAG 시스템 준비 완료! 이제 더 정확한 상담이 가능해요 ✅")

# 탭 생성
tab1, tab2 = st.tabs(["💬 건강 고민", "🍽️ 음식 이미지 분석"])

with tab1:
    st.subheader("건강 고민 상담")
    st.write("다이어트나 건강 관련 고민을 편하게 말씀해주세요 😊")
    
    concern = st.text_area(
        "고민을 입력해주세요:", 
        placeholder="예: 야식을 끊고 싶은데 자꾸 생각나요...",
        height=100
    )
    
    if st.button("🩺 우창윤 선생님께 상담받기", type="primary"):
        if concern:
            with st.spinner("선생님이 관련 의료 지식을 찾고 답변을 준비하고 있어요... ⏰"):
                # RAG: 관련 의료 지식 검색
                relevant_knowledge = get_relevant_knowledge(concern, vectorstore)
                
                # 검색된 지식 표시 (디버깅용 - 실제 서비스에서는 제거 가능)
                with st.expander("🔍 검색된 관련 의료 지식 (RAG 결과)"):
                    st.write(relevant_knowledge)
                
                # 우창윤 스타일 상담 프롬프트 생성
                consultation_prompt = create_woo_style_consultation_prompt(concern, relevant_knowledge)
                
                response = llm.invoke(consultation_prompt)
                
                st.success("상담 완료! 📋")
                st.write("**💬 우창윤 선생님의 따뜻한 상담:**")
                
                # 답변을 예쁘게 표시
                with st.container():
                    st.markdown("---")
                    st.markdown(f"""
                    <div style='background-color: black; padding: 20px; border-radius: 10px; border-left: 4px solid #4CAF50; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                        <div style='display: flex; align-items: center; margin-bottom: 10px;'>
                            <span style='font-size: 20px; margin-right: 8px;'>👨‍⚕️</span>
                            <strong style='color: black !important;'>우창윤 선생님</strong>
                        </div>
                        <p style='margin: 0; line-height: 1.6; color: black !important;'>{response.content}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("---")
        else:
            st.warning("고민을 입력해주세요!")

with tab2:
    st.subheader("음식 이미지 분석 🔍")
    st.write("음식 사진을 올려주시면 우창윤 선생님이 RAG 시스템으로 더 정확하게 분석해드려요!")
    
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
        
        # 우창윤 선생님 RAG 분석 버튼
        if st.button("🩺 우창윤 선생님 음식 상담받기", type="primary", use_container_width=True):
            with st.spinner("우창윤 선생님이 이미지를 분석하고 관련 의료 지식을 찾고 있어요... 🤖📚"):
                try:
                    # 1단계: 음식명 인식
                    base64_image = encode_image_to_base64(image)
                    
                    recognition_message = HumanMessage(
                        content=[
                            {
                                "type": "text",
                                "text": "이 이미지에 있는 음식의 이름을 '음식이름(정확한 칼로리)' 형식으로 정확하게 알려주세요. '음식명(몇 칼로리)' 형식으로만 답변해주세요. 한국어로 답변해주세요."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    )
                    
                    food_recognition = llm.invoke([recognition_message])
                    recognized_food = food_recognition.content.strip()
                    
                    st.info(f"🔍 **인식된 음식:** {recognized_food}")
                    
                    # 2단계: RAG 검색 - 인식된 음식에 대한 관련 의료 지식 검색
                    relevant_knowledge = get_relevant_knowledge(recognized_food, vectorstore, k=3)
                    
                    # 검색된 지식 표시 (디버깅용)
                    with st.expander("🔍 검색된 관련 의료 지식 (RAG 결과)"):
                        st.write(relevant_knowledge)
                    
                    # 3단계: 우창윤 선생님 스타일 분석 생성
                    image_analysis_prompt = f"""
                    당신은 내분비내과 전문의 우창윤 선생님입니다. 다음은 우창윤 선생님의 실제 말투 예시들입니다:

                    실제 말투 예시:
                    - "안녕하세요, 위머님! 전 내분비내과 전문의로 비만과 대사 질환, 당뇨병에 관심이 많아요."
                    - "팥빙수에 연유, 꼭 넣어야 할까요? 한 번쯤 안 넣고 드셔보는 건 어떨까요?"
                    - "그렇지 않답니다 ㅎ 조금만 더 버텨보세요."
                    
                    인식된 음식: {recognized_food}

                    관련 의료 지식 (RAG 검색 결과):
                    {relevant_knowledge}

                    **중요**: 반드시 우창윤 선생님의 친근한 말투로만 답변하세요!

                    핵심 말투 요소 (필수 포함):
                    - "안녕하세요, 위머님!" 같은 친근한 인사로 시작
                    - "~죠", "~답니다", "~해보는 건 어떨까요?" 어미 사용
                    - 의료 지식에 기반한 솔루션 제공
                    - 친근하고 부드러운 톤 유지

                    답변 형태:
                    "안녕하세요, 위머님! [음식명]을/를 보니 [첫인상/특징]죠 
                    
                    이 음식은 [칼로리 수치]정도 되고 [RAG 지식 기반 영양학적 분석]라서 [건강에 미치는 영향]답니다~ 
                    [구체적 이유]때문에 [주의사항]해요. 
                    대신 [대안음식/개선방법]으로 바꿔보는 건 어떨까요? 
                    괜찮습니다, 조금만 더 신경써주시면 분명 좋아질 거예요!"

                    반드시 우창윤 선생님의 따뜻하고 친근한 말투로 음식을 분석해주세요.
                    """
                    
                    feedback_response = llm.invoke(image_analysis_prompt)
                    
                    # 최종 결과 표시
                    st.success("RAG 분석 완료! 🎯")
                    st.write("**💬 우창윤 선생님의 친근한 피드백:**")
                    
                    # 답변을 예쁘게 표시
                    with st.container():
                        st.markdown("---")
                        st.markdown(f"""
                        <div style='background-color: black; padding: 20px; border-radius: 10px; border-left: 4px solid #4CAF50; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                            <div style='display: flex; align-items: center; margin-bottom: 10px;'>
                                <span style='font-size: 20px; margin-right: 8px;'>👨‍⚕️</span>
                                <strong style='color: black !important;'>우창윤 선생님 (RAG 분석)</strong>
                            </div>
                            <p style='margin: 0; line-height: 1.6; color: black !important;'>{feedback_response.content}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown("---")
                    
                except Exception as e:
                    st.error(f"분석 중 오류가 발생했습니다: {str(e)}")
                    st.write("다시 시도해보시거나 다른 이미지를 업로드해주세요.")

# 사이드바에 RAG 정보 및 사용법 안내
with st.sidebar:
    st.markdown("""
    ## 👨‍⚕️ 우창윤 선생님 소개
    
    안녕하세요! 내분비내과 전문의로 
    비만과 대사 질환, 당뇨병에 관심이 많아요 😊
    
    ## 🧠 RAG 시스템 정보
    
    **현재 의료 지식 베이스:**
    - 📊 총 지식 문서: 10개
    - 🔍 벡터 검색: FAISS
    - 📝 임베딩: OpenAI
    - 🎯 관련도 기반 지식 검색
    
    **포함된 의료 지식:**
    - 탄수화물과 혈당 관리
    - 단백질과 근육량 유지  
    - 지방 섭취와 심혈관 건강
    - 나트륨과 혈압 관리
    - 식이섬유의 중요성
    - 패스트푸드의 문제점
    - 음료와 건강 관리
    
    ## 📖 사용법
    
    ### 💬 건강 고민 탭
    - 고민 입력 시 RAG가 관련 의료 지식을 자동 검색
    - 더 정확하고 전문적인 상담 제공
    
    ### 🍽️ 음식 이미지 분석 탭
    1. 📸 음식 사진 업로드
    2. 🤖 AI가 음식명 자동 인식
    3. 🔍 RAG가 관련 의료 지식 검색
    4. 🩺 우창윤 선생님 스타일 분석 제공
    
    ---
    
    💡 **RAG의 장점**: 
    - 🎯 정확한 관련 지식 검색
    - 📚 체계적인 의료 정보 활용
    - 🔄 지속적인 지식 업데이트 가능
    """)

# 메인 페이지 하단에 RAG 시스템 정보
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🧠 <strong>RAG 기반 AI 시스템</strong>으로 더 정확한 건강 상담을 제공합니다!</p>
    <p>💙 건강한 식단으로 더 나은 내일을 만들어가요!</p>
    <p>궁금한 점이 있으시면 언제든 상담받아보세요 😊</p>
</div>
""", unsafe_allow_html=True)