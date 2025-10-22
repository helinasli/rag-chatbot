"""
RAG (Retrieval-Augmented Generation) Sistemi
============================================
Bu modül, doküman retrieval ve text generation işlemlerini gerçekleştirir.

RAG Yaklaşımı:
1. Dokümanları vektör veritabanında saklar
2. Kullanıcı sorusunu vektörize eder
3. En benzer dokümanları bulur (semantic search)
4. Bulunan dokümanları bağlam olarak kullanıp LLM ile yanıt üretir

Temel Bileşenler:
- Embedding: Sentence Transformers (all-MiniLM-L6-v2)
- Vektör DB: ChromaDB
- LLM: Google Gemini Pro
"""

import os
import logging
from typing import List, Dict, Any
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv

# Environment variables'ları .env dosyasından yükle
load_dotenv()

# Logging konfigürasyonu - bilgilendirme mesajları için
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    """
    RAG (Retrieval-Augmented Generation) sistemi sınıfı.
    Dokümanları vektörize eder, benzer dokümanları bulur ve yanıt üretir.
    """
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        RAG sistemini başlatır.
        
        Bu method tüm RAG bileşenlerini initialize eder:
        1. Embedding model'i yükler (metinleri vektörlere dönüştürmek için)
        2. Gemini API'yi konfigüre eder (yanıt üretmek için)
        3. ChromaDB vektör veritabanını hazırlar (doküman saklamak için)
        
        Args:
            embedding_model_name (str): Kullanılacak Sentence Transformer model adı
                                       Varsayılan: "all-MiniLM-L6-v2" (384 boyutlu vektörler)
        """
        # Embedding model'i yükle - metinleri sayısal vektörlere dönüştürmek için
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Veritabanı ve koleksiyon değişkenlerini başlat
        self.chroma_client = None
        self.collection = None
        
        # Gemini API'yi konfigüre et (text generation için)
        self.setup_gemini()
        
        # ChromaDB vektör veritabanını hazırla
        self.setup_vector_db()
        
    def setup_gemini(self):
        """
        Gemini API'yi konfigüre eder ve kullanılabilir bir model seçer.
        
        Bu method:
        1. Environment'tan API key'i alır
        2. Mevcut modelleri kontrol eder
        3. Çalışan bir model bulana kadar farklı model isimlerini dener
        4. Test ile model çalışırlığını doğrular
        
        API key bulunamazsa veya model yüklenemezse self.model = None olur.
        """
        # API key'i environment variable'dan al
        api_key = os.getenv("GEMINI_API_KEY")
        
        # API key kontrolü
        if not api_key or api_key == "your_gemini_api_key_here":
            logger.warning("GEMINI_API_KEY bulunamadı veya geçersiz. .env dosyasını kontrol edin.")
            self.model = None
            return
        
        try:
            # Gemini API'yi yapılandır
            genai.configure(api_key=api_key)
            
            # Mevcut modelleri listele (opsiyonel - debugging için)
            try:
                models = genai.list_models()
                # Sadece generateContent destekleyen modelleri filtrele
                available_models = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
                logger.info(f"Mevcut modeller: {available_models[:3]}...")  # İlk 3'ünü göster
            except Exception as e:
                logger.warning(f"Model listesi alınamadı: {e}")
                available_models = []
            
            # Denenmesi gereken model isimleri (yeniden eskiye doğru)
            # Gemini API'de model isimleri sık değişebilir, bu nedenle birden fazla seçenek deniyoruz
            model_names_to_try = [
                'gemini-2.5-flash',                 # En yeni model
                'gemini-2.5-flash-preview-05-20',
                'gemini-2.5-pro-preview-03-25',
                'gemini-1.5-pro-latest',
                'gemini-1.5-pro', 
                'gemini-1.5-flash-latest',
                'gemini-1.5-flash',
                'gemini-pro'                        # Eski stable model
            ]
            
            # Çalışan bir model bulana kadar dene
            self.model = None
            for model_name in model_names_to_try:
                try:
                    self.model = genai.GenerativeModel(model_name)
                    # Basit bir test ile model'in çalıştığını doğrula
                    test_response = self.model.generate_content("Test")
                    logger.info(f"Gemini API başarıyla konfigüre edildi ({model_name}).")
                    break  # Çalışan model bulundu, döngüden çık
                except Exception as e:
                    logger.debug(f"{model_name} modeli denenemedi: {e}")
                    continue  # Bu model çalışmadı, sıradakini dene
            
            # Hiçbir model çalışmadıysa hata logla
            if self.model is None:
                logger.error("Hiçbir Gemini model çalışmadı. API key'inizi kontrol edin.")
                
        except Exception as e:
            logger.error(f"Gemini API konfigürasyon hatası: {e}")
            self.model = None
    
    def setup_vector_db(self):
        """
        ChromaDB vektör veritabanını kurar.
        
        ChromaDB, yüksek boyutlu vektörleri saklamak ve benzerlik araması yapmak için kullanılır.
        Bu method:
        1. Persistent client oluşturur (veriler ./chroma_db klasöründe saklanır)
        2. "documents" adlı bir koleksiyon oluşturur veya mevcutu yükler
        3. Cosine similarity metriğini kullanacak şekilde yapılandırır
        
        Raises:
            Exception: Veritabanı kurulamazsa hata fırlatır
        """
        try:
            # ChromaDB client'ı başlat
            # PersistentClient kullanarak veriler diske kaydedilir ve kalıcı olur
            self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
            
            # Koleksiyon oluştur veya mevcut olanı al
            try:
                # Yeni koleksiyon oluştur
                self.collection = self.chroma_client.create_collection(
                    name="documents",
                    metadata={"hnsw:space": "cosine"}  # Cosine similarity kullan
                )
                logger.info("Yeni ChromaDB koleksiyonu oluşturuldu.")
            except Exception:
                # Koleksiyon zaten varsa mevcut olanı kullan
                self.collection = self.chroma_client.get_collection("documents")
                logger.info("Mevcut ChromaDB koleksiyonu yüklendi.")
                
        except Exception as e:
            logger.error(f"ChromaDB kurulumu hatası: {e}")
            raise
    
    def load_and_process_data(self, data_path: str = None, data_dict: Dict = None, hf_dataset: str = None):
        """
        Veri yükler ve işler, ardından vektör veritabanına ekler.
        
        Bu method üç farklı veri kaynağını destekler:
        1. data_dict: Python dictionary (key: başlık, value: içerik)
        2. data_path: CSV dosyası (ilk sütun metin olarak kullanılır)
        3. hf_dataset: Hugging Face dataset adı
        
        Hiçbiri verilmezse varsayılan örnek veriler kullanılır.
        
        İşlem Adımları:
        1. Veri kaynağından dokümanları yükle
        2. Her doküman için embedding oluştur (vektörize et)
        3. Embedding'leri metadata ile birlikte ChromaDB'ye kaydet
        
        Args:
            data_path (str, optional): CSV dosya yolu
            data_dict (Dict, optional): Doğrudan veri dictionary'si
            hf_dataset (str, optional): Hugging Face dataset adı
        """
        # Seçenek 1: Dictionary'den doğrudan veri al
        if data_dict:
            # Dictionary formatı: {"başlık": "içerik", ...}
            # Her key-value çiftini bir doküman olarak işle
            documents = []
            metadatas = []
            ids = []
            
            for i, (key, value) in enumerate(data_dict.items()):
                documents.append(str(value))  # İçeriği doküman olarak ekle
                metadatas.append({"source": key, "id": i})  # Metadata: başlık ve id
                ids.append(f"doc_{i}")  # Benzersiz doküman ID'si
                
        # Seçenek 2: Hugging Face dataset'ten yükle
        elif hf_dataset:
            # Hugging Face dataset'ten yükle
            try:
                from datasets import load_dataset
                
                # Dataset'i yükle
                logger.info(f"Hugging Face dataset yükleniyor: {hf_dataset}")
                # Script-based dataset'ler için farklı yöntem dene
                try:
                    dataset = load_dataset(hf_dataset, split="train")
                except Exception as e:
                    logger.warning(f"İlk deneme başarısız: {e}")
                    # Parquet/Arrow formatını dene
                    dataset = load_dataset(hf_dataset, split="train", streaming=False)
                
                # Büyük dataset'lerde sadece ilk N örneği al
                max_documents = 5000  # Maksimum doküman sayısı
                if len(dataset) > max_documents:
                    logger.info(f"Dataset çok büyük ({len(dataset)} örnek), ilk {max_documents} örnek alınıyor...")
                    dataset = dataset.select(range(max_documents))
                
                # İlk text/content field'ı bul
                text_column = None
                for col in dataset.column_names:
                    if 'text' in col.lower() or 'content' in col.lower() or 'description' in col.lower():
                        text_column = col
                        break
                
                if text_column:
                    documents = [str(item[text_column]) for item in dataset]
                    logger.info(f"'{text_column}' sütunu kullanılıyor")
                else:
                    # İlk string column'ı kullan
                    documents = [str(item[dataset.column_names[0]]) for item in dataset]
                    logger.info(f"'{dataset.column_names[0]}' sütunu kullanılıyor")
                
                metadatas = [{"source": "huggingface", "dataset": hf_dataset, "id": i} for i in range(len(documents))]
                ids = [f"hf_doc_{i}" for i in range(len(documents))]
                
            except Exception as e:
                logger.error(f"Hugging Face dataset yükleme hatası: {e}")
                documents = []
                metadatas = []
                ids = []
                
        # Seçenek 3: CSV dosyasından yükle
        elif data_path and os.path.exists(data_path):
            # Pandas ile CSV dosyasını oku
            df = pd.read_csv(data_path)
            # İlk sütunu doküman olarak kullan (tüm satırları string'e çevir)
            documents = df.iloc[:, 0].astype(str).tolist()
            metadatas = [{"source": "csv", "id": i} for i in range(len(documents))]
            ids = [f"doc_{i}" for i in range(len(documents))]
        
        # Seçenek 4: Varsayılan örnek veri kullan
        else:
            # Hiçbir veri kaynağı belirtilmediyse varsayılan örnekleri yükle
            # Bu dokümanlar AI, ML, DL konularında genel bilgiler içerir
            documents = [
                "Yapay zeka, makinelerin insan benzeri düşünme yeteneklerini simüle etmesidir.",
                "Makine öğrenmesi, yapay zekanın bir alt dalıdır ve verilerden öğrenmeyi sağlar.",
                "Derin öğrenme, çok katmanlı sinir ağları kullanarak karmaşık desenleri öğrenir.",
                "Doğal dil işleme, bilgisayarların insan dilini anlamasını ve işlemesini sağlar.",
                "RAG sistemi, bilgi retrieval ve text generation'ı birleştirir.",
                "Vektör veritabanları, yüksek boyutlu vektörleri verimli şekilde saklar.",
                "Embedding modelleri, metinleri sayısal vektörlere dönüştürür.",
                "Transformer mimarisi, modern NLP modellerinin temelini oluşturur."
            ]
            metadatas = [{"source": "default", "id": i} for i in range(len(documents))]
            ids = [f"doc_{i}" for i in range(len(documents))]
        
        # Tüm dokümanlar için embeddings (vektör temsilleri) oluştur
        # Sentence Transformer her metni 384 boyutlu bir vektöre dönüştürür
        logger.info("Dokümanlar için embeddings oluşturuluyor...")
        embeddings = self.embedding_model.encode(documents).tolist()
        
        # Dokümanları, embedding'leri ve metadata'ları ChromaDB'ye ekle
        # Bu sayede daha sonra benzerlik araması yapabiliriz
        self.collection.add(
            documents=documents,        # Orijinal metin içeriği
            embeddings=embeddings,      # Vektör temsilleri (384 boyutlu)
            metadatas=metadatas,        # Ek bilgiler (kaynak, id)
            ids=ids                     # Benzersiz doküman kimlikleri
        )
        
        logger.info(f"{len(documents)} doküman başarıyla yüklendi ve vektörize edildi.")
    
    def retrieve_documents(self, query: str, n_results: int = 3) -> List[Dict]:
        """
        Sorguya en benzer dokümanları semantic search ile getirir.
        
        Bu method RAG sisteminin "Retrieval" kısmıdır:
        1. Kullanıcı sorgusunu vektöre dönüştürür
        2. ChromaDB'de cosine similarity ile en benzer dokümanları bulur
        3. En yüksek benzerlik skoruna sahip n_results kadar doküman döndürür
        
        Args:
            query (str): Kullanıcı sorgusu (doğal dil metni)
            n_results (int): Getirilecek doküman sayısı (varsayılan: 3)
            
        Returns:
            List[Dict]: Benzer dokümanlar listesi, her biri şunları içerir:
                - content: Doküman metni
                - metadata: Doküman hakkında ek bilgiler
                - distance: Benzerlik skoru (düşük = daha benzer)
        """
        if not self.collection:
            logger.error("Koleksiyon bulunamadı. Önce veri yükleyin.")
            return []
        
        # Kullanıcı sorgusunu embedding'e dönüştür (vektörize et)
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        # ChromaDB'de benzerlik araması yap
        # Cosine similarity kullanarak en yakın vektörleri bul
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        # Sonuçları düzenle ve liste formatında döndür
        retrieved_docs = []
        if results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                retrieved_docs.append({
                    'content': doc,  # Doküman içeriği
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else 0  # Benzerlik skoru
                })
        
        return retrieved_docs
    
    def generate_response(self, query: str, retrieved_docs: List[Dict]) -> str:
        """
        Retrieval edilen dokümanları kullanarak LLM ile yanıt üretir.
        
        Bu method RAG sisteminin "Generation" kısmıdır:
        1. Retrieval edilen dokümanları birleştirip context oluşturur
        2. Context ve query'yi içeren bir prompt hazırlar
        3. Gemini LLM'den Türkçe yanıt üretmesini ister
        4. Yanıtı döndürür
        
        Prompt Engineering: LLM'e sadece verilen bağlam içinde yanıt vermesi,
        Türkçe yanıt üretmesi talimatı verilir.
        
        Args:
            query (str): Kullanıcı sorgusu
            retrieved_docs (List[Dict]): Retrieval edilen dokümanlar
            
        Returns:
            str: LLM tarafından üretilen yanıt metni
        """
        # Model kontrolü
        if not hasattr(self, 'model') or self.model is None:
            return "Gemini API konfigüre edilmemiş veya model yüklenemedi. Lütfen GEMINI_API_KEY'i kontrol edin ve geçerli bir API key kullandığınızdan emin olun."
        
        # Retrieval edilen dokümanları birleştirip tek bir context metni oluştur
        # Her doküman arasına iki satır boşluk ekle
        context = "\n\n".join([doc['content'] for doc in retrieved_docs])
        
        # LLM için prompt hazırla (Prompt Engineering)
        # Bu prompt, LLM'e ne yapması gerektiğini açıkça söyler
        prompt = f"""
        Aşağıdaki bağlam bilgilerini kullanarak soruyu yanıtla. Yanıtın Türkçe olsun ve bağlam bilgilerine dayalı olsun.
        
        Bağlam:
        {context}
        
        Soru: {query}
        
        Yanıt:
        """
        
        try:
            # Gemini LLM'den yanıt al
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Yanıt üretme hatası: {e}")
            return f"Yanıt üretilirken hata oluştu: {e}"
    
    def chat(self, query: str) -> Dict[str, Any]:
        """
        Tam RAG pipeline'ını çalıştırır (end-to-end).
        
        Bu method tüm RAG sürecini yönetir:
        1. Kullanıcı sorgusunu alır
        2. Retrieval: En benzer dokümanları bulur (retrieve_documents)
        3. Generation: Bulunan dokümanları kullanarak yanıt üretir (generate_response)
        4. Sonuçları ve metadata'yı döndürür
        
        Bu, kullanıcı ile etkileşim için ana method'dur.
        
        Args:
            query (str): Kullanıcının sorduğu soru (doğal dil)
            
        Returns:
            Dict[str, Any]: Şunları içeren dictionary:
                - query: Kullanıcı sorgusu
                - response: LLM tarafından üretilen yanıt
                - retrieved_documents: Kullanılan kaynak dokümanlar
                - num_retrieved: Kaç doküman kullanıldığı
        """
        # 1. RETRIEVAL: Sorguya en benzer dokümanları bul
        retrieved_docs = self.retrieve_documents(query)
        
        # 2. GENERATION: Bulunan dokümanları kullanarak yanıt üret
        response = self.generate_response(query, retrieved_docs)
        
        # 3. Sonuçları ve metadata'yı döndür
        return {
            'query': query,                          # Orijinal soru
            'response': response,                    # LLM'den gelen yanıt
            'retrieved_documents': retrieved_docs,   # Kullanılan kaynak dokümanlar
            'num_retrieved': len(retrieved_docs)     # Kaç doküman kullanıldı
        }

if __name__ == "__main__":
    # Test kodu
    rag = RAGSystem()
    rag.load_and_process_data()
    
    test_query = "Yapay zeka nedir?"
    result = rag.chat(test_query)
    
    print(f"Soru: {result['query']}")
    print(f"Yanıt: {result['response']}")
    print(f"Kullanılan doküman sayısı: {result['num_retrieved']}")
