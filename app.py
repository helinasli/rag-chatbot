"""
RAG Chatbot Web Arayüzü
Streamlit kullanarak RAG sisteminin web arayüzü
"""

import streamlit as st
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Src klasörünü path'e ekle
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from rag_system import RAGSystem
except ImportError as e:
    st.error(f"RAG sistemi yüklenemedi: {e}")
    st.stop()

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="RAG Chatbot",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS stilleri - Profesyonel tasarım
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: left;
        margin-top: -2rem;
        margin-bottom: 2rem;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    .chat-message {
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 3px solid #34495e;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .user-message {
        background-color: #ecf0f1;
        border-left-color: #3498db;
    }
    .bot-message {
        background-color: #f8f9fa;
        border-left-color: #2c3e50;
    }
    .stButton > button {
        background-color: #2c3e50;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stButton > button:hover {
        background-color: #34495e;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_rag_system():
    """RAG sistemini başlatır ve cache'ler."""
    try:
        rag = RAGSystem()
        return rag
    except Exception as e:
        st.error(f"RAG sistemi başlatılamadı: {e}")
        return None

def load_sample_data():
    """Örnek veri seti yükler."""
    sample_data = {
        "Yapay Zeka Temelleri": "Yapay zeka (AI), makinelerin insan benzeri düşünme, öğrenme ve problem çözme yeteneklerini simüle etmesidir. AI sistemleri, veri analizi, pattern recognition ve karar verme süreçlerinde kullanılır.",
        
        "Makine Öğrenmesi": "Makine öğrenmesi, yapay zekanın bir alt dalıdır ve algoritmaların deneyimlerden öğrenmesini sağlar. Supervised, unsupervised ve reinforcement learning olmak üzere üç ana kategoriye ayrılır.",
        
        "Derin Öğrenme": "Derin öğrenme, çok katmanlı sinir ağları kullanarak karmaşık desenleri öğrenir. Görüntü işleme, doğal dil işleme ve ses tanıma gibi alanlarda devrim yaratmıştır.",
        
        "Doğal Dil İşleme": "NLP, bilgisayarların insan dilini anlaması, işlemesi ve üretmesini sağlayan AI dalıdır. Metin analizi, çeviri, sentiment analizi gibi uygulamalarda kullanılır.",
        
        "RAG Sistemi": "Retrieval-Augmented Generation, bilgi retrieval ve text generation'ı birleştiren bir yaklaşımdır. Büyük dil modellerinin bilgi tabanından faydalanarak daha doğru yanıtlar üretmesini sağlar.",
        
        "Vektör Veritabanları": "Vektör veritabanları, yüksek boyutlu vektörleri verimli şekilde saklar ve benzerlik araması yapar. ChromaDB, Pinecone ve FAISS gibi çözümler popülerdir.",
        
        "Transformer Mimarisi": "Transformer, attention mechanism kullanan ve modern NLP modellerinin temelini oluşturan sinir ağı mimarisidir. BERT, GPT ve T5 gibi modeller bu mimariye dayanır.",
        
        "Embedding Modelleri": "Embedding modelleri, metinleri sayısal vektörlere dönüştürür. Bu vektörler, semantic similarity hesaplama ve information retrieval için kullanılır."
    }
    return sample_data

def main():
    """Ana uygulama fonksiyonu."""
    
    # Başlık
    st.markdown('<h1 class="main-header">RAG Tabanlı Chatbot</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Proje Bilgileri")
        st.markdown("""
        **Teknolojiler:**
        - Retrieval: ChromaDB + Sentence Transformers
        - Generation: Google Gemini API
        - Interface: Streamlit
        - Framework: LangChain
        """)
        
        # API Key durumu (sadece gerektiğinde göster)
        load_dotenv()
        
        # Streamlit secrets'tan veya .env'den API key al
        existing_api_key = None
        if hasattr(st, 'secrets') and 'GEMINI_API_KEY' in st.secrets:
            existing_api_key = st.secrets["GEMINI_API_KEY"]
            os.environ["GEMINI_API_KEY"] = existing_api_key
        else:
            existing_api_key = os.getenv("GEMINI_API_KEY")
        
        if not existing_api_key or existing_api_key == "your_gemini_api_key_here":
            st.markdown("### ⚙️ API Konfigürasyonu")
            st.warning("⚠️ Gemini API key bulunamadı")
            st.info("💡 Lokal kullanım için .env dosyasına ekleyin veya aşağıya girin")
            api_key = st.text_input("Gemini API Key", type="password", help="Google AI Studio'dan alabilirsiniz: https://ai.google.dev/")
            
            if api_key:
                os.environ["GEMINI_API_KEY"] = api_key
                st.success("✅ API Key ayarlandı!")
        else:
            st.sidebar.success("✅ API Key aktif")
        
        # Veri yükleme seçenekleri
        st.markdown("### Veri Yükleme")
        data_option = st.selectbox(
            "Veri kaynağı seçin:",
            ["Örnek Veri Kullan", "CSV Dosyası Yükle", "Manuel Veri Girişi"]
        )
        
        uploaded_file = None
        manual_data = None
        
        if data_option == "CSV Dosyası Yükle":
            uploaded_file = st.file_uploader("CSV dosyası seçin", type=['csv'])
        elif data_option == "Manuel Veri Girişi":
            manual_data = st.text_area("Veri girin (her satır bir doküman)", height=150)
    
    # Ana içerik alanı
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Sohbet Alanı")
        
        # Chat history için session state
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        if "rag_system" not in st.session_state:
            st.session_state.rag_system = None
        
        # RAG sistemi başlatma
        if st.button("RAG Sistemini Başlat", type="primary"):
            with st.spinner("RAG sistemi başlatılıyor..."):
                rag_system = initialize_rag_system()
                
                if rag_system:
                    # Veri yükleme
                    if data_option == "Örnek Veri Kullan":
                        sample_data = load_sample_data()
                        rag_system.load_and_process_data(data_dict=sample_data)
                        st.success("Örnek veri yüklendi!")
                    
                    elif data_option == "CSV Dosyası Yükle" and uploaded_file:
                        # CSV dosyasını geçici olarak kaydet
                        temp_path = f"temp_{uploaded_file.name}"
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        rag_system.load_and_process_data(data_path=temp_path)
                        os.remove(temp_path)  # Geçici dosyayı sil
                        st.success("CSV dosyası yüklendi!")
                    
                    elif data_option == "Manuel Veri Girişi" and manual_data:
                        lines = manual_data.strip().split('\n')
                        manual_dict = {f"doc_{i}": line for i, line in enumerate(lines) if line.strip()}
                        rag_system.load_and_process_data(data_dict=manual_dict)
                        st.success("Manuel veri yüklendi!")
                    
                    else:
                        # Varsayılan veri yükle
                        rag_system.load_and_process_data()
                        st.success("Varsayılan veri yüklendi!")
                    
                    st.session_state.rag_system = rag_system
        
        # Chat arayüzü
        if st.session_state.rag_system:
            # Mesaj geçmişini göster
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f'<div class="chat-message user-message"><strong>Siz:</strong> {message["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message bot-message"><strong>Bot:</strong> {message["content"]}</div>', unsafe_allow_html=True)
            
            # Yeni mesaj girişi
            user_input = st.text_input("Sorunuzu yazın:", key="user_input")
            
            if st.button("Gönder") and user_input:
                # Kullanıcı mesajını ekle
                st.session_state.messages.append({"role": "user", "content": user_input})
                
                # RAG sisteminden yanıt al
                with st.spinner("Yanıt üretiliyor..."):
                    try:
                        result = st.session_state.rag_system.chat(user_input)
                        bot_response = result['response']
                        
                        # Bot yanıtını ekle
                        st.session_state.messages.append({"role": "assistant", "content": bot_response})
                        
                        # Sayfayı yenile
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Hata oluştu: {e}")
        
        else:
            st.info("RAG sistemini başlatmak için yukarıdaki butona tıklayın.")
    
    with col2:
        st.markdown("### Sistem Durumu")
        
        if st.session_state.rag_system:
            st.success("RAG Sistemi Aktif")
            
            # İstatistikler
            if hasattr(st.session_state.rag_system, 'collection') and st.session_state.rag_system.collection:
                try:
                    doc_count = st.session_state.rag_system.collection.count()
                    st.metric("Yüklü Doküman", doc_count)
                except:
                    st.metric("Yüklü Doküman", "Bilinmiyor")
            
            st.metric("Toplam Mesaj", len(st.session_state.messages))
            
            # Son güncelleme zamanı
            st.info(f"Son Güncelleme: {datetime.now().strftime('%H:%M:%S')}")
            
        else:
            st.warning("RAG Sistemi Pasif")
        
        # Kullanım kılavuzu
        st.markdown("### Kullanım Kılavuzu")
        st.markdown("""
        1. **Veri**: Veri kaynağınızı seçin
        2. **Başlat**: RAG sistemini başlatın
        3. **Sohbet**: Sorularınızı yazın!
        
        **Örnek Sorular:**
        - "Yapay zeka nedir?"
        - "RAG sistemi nasıl çalışır?"
        - "Makine öğrenmesi türleri neler?"
        """)
        
        # Temizleme butonu
        if st.button("Sohbeti Temizle"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()
