"""
RAG Chatbot Web Arayüzü
=======================
Bu modül, Streamlit kullanarak RAG (Retrieval-Augmented Generation) sisteminin
web arayüzünü sağlar. Kullanıcılar dokümanlar yükleyebilir ve chatbot ile etkileşime geçebilir.

Temel Özellikler:
- Gemini API entegrasyonu
- ChromaDB ile vektör araması
- CSV dosya yükleme desteği
- Real-time chat arayüzü
"""

import streamlit as st
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Src klasörünü Python path'ine ekle
# Bu sayede rag_system modülünü import edebiliriz
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
    """
    RAG sistemini başlatır ve cache'ler.
    
    Bu fonksiyon @st.cache_resource decorator'ı ile işaretlenmiştir.
    Bu sayede RAG sistemi bir kez oluşturulur ve tekrar kullanılır,
    böylece her sayfa yenilemede yeniden yüklenmez.
    
    Returns:
        RAGSystem: Başlatılmış RAG sistem nesnesi veya hata durumunda None
    """
    try:
        # RAG sistemini başlat (embedding model ve vektör DB dahil)
        rag = RAGSystem()
        return rag
    except Exception as e:
        st.error(f"RAG sistemi başlatılamadı: {e}")
        return None

def load_sample_data():
    """
    Örnek veri seti yükler.
    
    Akbank ve finans konulu örnek dokümanları içeren bir dictionary döndürür.
    Bu dokümanlar RAG sistemine yüklenip vektörize edilir.
    
    Returns:
        dict: Başlık-içerik çiftlerinden oluşan doküman sözlüğü
    """
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
    """
    Ana uygulama fonksiyonu.
    
    Bu fonksiyon uygulamanın ana akışını yönetir:
    1. Sayfa başlığını ve sidebar'ı oluşturur
    2. API key konfigürasyonunu kontrol eder
    3. Veri yükleme seçeneklerini sunar
    4. Chat arayüzünü ve mesaj geçmişini yönetir
    5. RAG sistemini kullanarak kullanıcı sorularına yanıt verir
    """
    
    # Sayfa başlığını göster
    st.markdown('<h1 class="main-header">RAG Tabanlı Chatbot</h1>', unsafe_allow_html=True)
    
    # Sidebar - Sol panel konfigürasyonu
    with st.sidebar:
        st.markdown("### Proje Bilgileri")
        st.markdown("""
        **Teknolojiler:**
        - Retrieval: ChromaDB + Sentence Transformers
        - Generation: Google Gemini API
        - Interface: Streamlit
        - Framework: LangChain
        """)
        
        # API Key durumu kontrol et (sadece gerektiğinde kullanıcıdan iste)
        load_dotenv()  # .env dosyasından environment variable'ları yükle
        
        # API key'i iki kaynaktan alabilir:
        # 1. Streamlit Cloud deployment için secrets
        # 2. Lokal kullanım için .env dosyası
        existing_api_key = None
        if hasattr(st, 'secrets') and 'GEMINI_API_KEY' in st.secrets:
            # Streamlit Cloud üzerinde secrets'tan al
            existing_api_key = st.secrets["GEMINI_API_KEY"]
            os.environ["GEMINI_API_KEY"] = existing_api_key
        else:
            # Lokal ortamda .env dosyasından al
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
        st.info("📚 Varsayılan: AI/ML Dataset\n\n100 Türkçe doküman (Yapay Zeka, Machine Learning, Deep Learning, NLP, RAG, vb.)")
        
        # Opsiyonel: Kullanıcı kendi CSV'sini de yükleyebilir
        use_custom_csv = st.checkbox("Kendi CSV dosyamı kullanmak istiyorum")
        
        uploaded_file = None
        if use_custom_csv:
            uploaded_file = st.file_uploader("CSV dosyası seçin", type=['csv'])
    
    # Ana içerik alanı - İki sütunlu düzen (2:1 oranında)
    # Sol sütun: Chat arayüzü, Sağ sütun: Durum bilgisi
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Sohbet Alanı")
        
        # Session state - Sayfalar arası veri paylaşımı için kullanılır
        # Streamlit her etkileşimde sayfayı yeniden çalıştırır,
        # bu nedenle state'i saklamak için session_state kullanılır
        if "messages" not in st.session_state:
            st.session_state.messages = []  # Chat geçmişini saklar
        
        if "rag_system" not in st.session_state:
            st.session_state.rag_system = None  # RAG sistemi instance'ını saklar
        
        # Otomatik başlatma - İlk açılışta RAG sistemini başlat
        if "initialized" not in st.session_state:
            st.session_state.initialized = False
        
        # RAG sistemi başlatma (otomatik veya manuel)
        if not st.session_state.initialized or st.button("RAG Sistemini Yeniden Başlat", type="secondary"):
            with st.spinner("RAG sistemi başlatılıyor..."):
                rag_system = initialize_rag_system()
                
                if rag_system:
                    # Veri yükleme işlemi
                    try:
                        if use_custom_csv and uploaded_file:
                            # Kullanıcının yüklediği CSV dosyasını işle
                            st.info("📄 Özel CSV dosyanız yükleniyor...")
                            temp_path = f"temp_{uploaded_file.name}"
                            with open(temp_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            
                            # CSV'yi RAG sistemine yükle ve vektörize et
                            rag_system.load_and_process_data(data_path=temp_path)
                            
                            # Geçici dosyayı temizle
                            os.remove(temp_path)
                            st.success("✅ CSV dosyası başarıyla yüklendi!")
                        
                        else:
                            # Varsayılan: Yerel AI/ML dataset'ini yükle
                            with st.spinner("📚 AI/ML Dataset yükleniyor..."):
                                rag_system.load_and_process_data(data_path="data/sample_data.csv")
                                st.success("✅ AI/ML Dataset başarıyla yüklendi!")
                        
                        # RAG sistemini session state'e kaydet
                        st.session_state.rag_system = rag_system
                        st.session_state.initialized = True  # Başlatma tamamlandı
                        
                    except Exception as e:
                        st.error(f"❌ Veri yüklenirken hata oluştu: {e}")
                        st.info("💡 İpucu: İnternet bağlantınızı kontrol edin veya özel CSV dosyası yüklemeyi deneyin")
                        st.session_state.initialized = True  # Hata olsa bile tekrar denemeyi engelle
        
        # Chat arayüzü - RAG sistemi aktifse göster
        if st.session_state.rag_system:
            # Eğer henüz mesaj yoksa, örnek sorular göster
            if len(st.session_state.messages) == 0:
                st.markdown("### 💡 Örnek Sorular:")
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("🤖 RAG sistemi nedir?", use_container_width=True):
                        st.session_state.messages.append({"role": "user", "content": "RAG sistemi nedir?"})
                        st.session_state.waiting_for_response = True
                        st.rerun()
                    if st.button("🧠 Transformer mimarisi nasıl çalışır?", use_container_width=True):
                        st.session_state.messages.append({"role": "user", "content": "Transformer mimarisi nasıl çalışır?"})
                        st.session_state.waiting_for_response = True
                        st.rerun()
                with col_b:
                    if st.button("💬 LLM nedir?", use_container_width=True):
                        st.session_state.messages.append({"role": "user", "content": "LLM nedir?"})
                        st.session_state.waiting_for_response = True
                        st.rerun()
                    if st.button("⚙️ Fine-tuning ne demek?", use_container_width=True):
                        st.session_state.messages.append({"role": "user", "content": "Fine-tuning ne demek?"})
                        st.session_state.waiting_for_response = True
                        st.rerun()
            
            # Tüm mesaj geçmişini ekranda göster
            # Her mesajı kullanıcı veya bot olarak farklı stillerle göster
            for message in st.session_state.messages:
                if message["role"] == "user":
                    # Kullanıcı mesajı - mavi kenarlık
                    st.markdown(f'<div class="chat-message user-message"><strong>Siz:</strong> {message["content"]}</div>', unsafe_allow_html=True)
                else:
                    # Bot mesajı - gri kenarlık
                    st.markdown(f'<div class="chat-message bot-message"><strong>Bot:</strong> {message["content"]}</div>', unsafe_allow_html=True)
            
            # Eğer yanıt bekleniyorsa, yanıt üret
            if st.session_state.get("waiting_for_response", False):
                with st.spinner("Yanıt üretiliyor..."):
                    try:
                        # En son kullanıcı mesajını al
                        last_user_message = st.session_state.messages[-1]["content"]
                        
                        # RAG pipeline'ını çalıştır:
                        # 1. Query'yi vektörize et
                        # 2. Benzer dokümanları bul (retrieval)
                        # 3. LLM ile yanıt üret (generation)
                        result = st.session_state.rag_system.chat(last_user_message)
                        bot_response = result['response']
                        
                        # Bot yanıtını geçmişe ekle
                        st.session_state.messages.append({"role": "assistant", "content": bot_response})
                        
                        # Waiting flag'i kaldır
                        st.session_state.waiting_for_response = False
                        
                        # Sayfayı yenile (yanıtı göstermek için)
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Hata oluştu: {e}")
                        st.session_state.waiting_for_response = False
            
            else:
                # Yanıt beklenmiyorsa, yeni mesaj giriş alanını göster
                # Form kullanarak (input otomatik temizlenir)
                with st.form(key="chat_form", clear_on_submit=True):
                    user_input = st.text_input("Sorunuzu yazın:", key="user_input_field")
                    submit_button = st.form_submit_button("Gönder")
                
                # Form submit edildiğinde
                if submit_button and user_input:
                    # Kullanıcı mesajını geçmişe ekle
                    st.session_state.messages.append({"role": "user", "content": user_input})
                    
                    # Waiting flag ekle - yanıt bekleniyor
                    st.session_state.waiting_for_response = True
                    
                    # Sayfayı hemen yenile (soruyu göstermek için)
                    st.rerun()
        
        else:
            # RAG sistemi henüz yüklenmediyse loading mesajı göster
            if not st.session_state.initialized:
                st.info("⏳ RAG sistemi otomatik olarak başlatılıyor... Lütfen bekleyin.")
            else:
                st.warning("⚠️ RAG sistemi yüklenemedi. Sayfayı yenileyin veya 'Yeniden Başlat' butonuna tıklayın.")
    
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
