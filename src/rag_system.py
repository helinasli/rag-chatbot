"""
RAG (Retrieval-Augmented Generation) Sistemi
Bu modül, doküman retrieval ve text generation işlemlerini gerçekleştirir.
"""

import os
import logging
from typing import List, Dict, Any
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv

# Environment variables yükleme
load_dotenv()

# Logging konfigürasyonu
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
        
        Args:
            embedding_model_name (str): Kullanılacak embedding model adı
        """
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.chroma_client = None
        self.collection = None
        self.setup_gemini()
        self.setup_vector_db()
        
    def setup_gemini(self):
        """Gemini API'yi konfigüre eder."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key == "your_gemini_api_key_here":
            logger.warning("GEMINI_API_KEY bulunamadı veya geçersiz. .env dosyasını kontrol edin.")
            self.model = None
            return
        
        try:
            genai.configure(api_key=api_key)
            
            # Mevcut modelleri listele
            try:
                models = genai.list_models()
                available_models = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
                logger.info(f"Mevcut modeller: {available_models[:3]}...")  # İlk 3'ünü göster
            except Exception as e:
                logger.warning(f"Model listesi alınamadı: {e}")
                available_models = []
            
            # Güncel model adlarını dene (API'den gelen listeden)
            model_names_to_try = [
                'gemini-2.5-flash',
                'gemini-2.5-flash-preview-05-20',
                'gemini-2.5-pro-preview-03-25',
                'gemini-1.5-pro-latest',
                'gemini-1.5-pro', 
                'gemini-1.5-flash-latest',
                'gemini-1.5-flash',
                'gemini-pro'
            ]
            
            self.model = None
            for model_name in model_names_to_try:
                try:
                    self.model = genai.GenerativeModel(model_name)
                    # Basit bir test yap
                    test_response = self.model.generate_content("Test")
                    logger.info(f"Gemini API başarıyla konfigüre edildi ({model_name}).")
                    break
                except Exception as e:
                    logger.debug(f"{model_name} modeli denenemedi: {e}")
                    continue
            
            if self.model is None:
                logger.error("Hiçbir Gemini model çalışmadı. API key'inizi kontrol edin.")
                
        except Exception as e:
            logger.error(f"Gemini API konfigürasyon hatası: {e}")
            self.model = None
    
    def setup_vector_db(self):
        """ChromaDB vektör veritabanını kurar."""
        try:
            # Yeni ChromaDB client konfigürasyonu
            self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
            
            # Koleksiyon oluştur veya mevcut olanı al
            try:
                self.collection = self.chroma_client.create_collection(
                    name="documents",
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info("Yeni ChromaDB koleksiyonu oluşturuldu.")
            except Exception:
                self.collection = self.chroma_client.get_collection("documents")
                logger.info("Mevcut ChromaDB koleksiyonu yüklendi.")
                
        except Exception as e:
            logger.error(f"ChromaDB kurulumu hatası: {e}")
            raise
    
    def load_and_process_data(self, data_path: str = None, data_dict: Dict = None, hf_dataset: str = None):
        """
        Veri yükler ve işler.
        
        Args:
            data_path (str): CSV dosya yolu
            data_dict (Dict): Doğrudan veri dictionary'si
            hf_dataset (str): Hugging Face dataset adı
        """
        if data_dict:
            # Doğrudan dictionary kullan
            documents = []
            metadatas = []
            ids = []
            
            for i, (key, value) in enumerate(data_dict.items()):
                documents.append(str(value))
                metadatas.append({"source": key, "id": i})
                ids.append(f"doc_{i}")
                
        elif hf_dataset:
            # Hugging Face dataset'ten yükle
            try:
                from datasets import load_dataset
                dataset = load_dataset(hf_dataset, split="train")
                # İlk text field'ı bul
                text_column = None
                for col in dataset.column_names:
                    if 'text' in col.lower() or 'content' in col.lower():
                        text_column = col
                        break
                
                if text_column:
                    documents = [str(item[text_column]) for item in dataset]
                else:
                    # İlk string column'ı kullan
                    documents = [str(item[dataset.column_names[0]]) for item in dataset]
                
                metadatas = [{"source": "huggingface", "dataset": hf_dataset, "id": i} for i in range(len(documents))]
                ids = [f"hf_doc_{i}" for i in range(len(documents))]
                
            except Exception as e:
                logger.error(f"Hugging Face dataset yükleme hatası: {e}")
                documents = []
                metadatas = []
                ids = []
                
        elif data_path and os.path.exists(data_path):
            # CSV dosyasından yükle
            df = pd.read_csv(data_path)
            documents = df.iloc[:, 0].astype(str).tolist()  # İlk sütunu al
            metadatas = [{"source": "csv", "id": i} for i in range(len(documents))]
            ids = [f"doc_{i}" for i in range(len(documents))]
        else:
            # Örnek veri oluştur
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
        
        # Embeddings oluştur
        logger.info("Dokümanlar için embeddings oluşturuluyor...")
        embeddings = self.embedding_model.encode(documents).tolist()
        
        # ChromaDB'ye ekle
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"{len(documents)} doküman başarıyla yüklendi ve vektörize edildi.")
    
    def retrieve_documents(self, query: str, n_results: int = 3) -> List[Dict]:
        """
        Sorguya en benzer dokümanları getirir.
        
        Args:
            query (str): Kullanıcı sorgusu
            n_results (int): Getirilecek doküman sayısı
            
        Returns:
            List[Dict]: Benzer dokümanlar listesi
        """
        if not self.collection:
            logger.error("Koleksiyon bulunamadı. Önce veri yükleyin.")
            return []
        
        # Query embedding oluştur
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        # Benzer dokümanları ara
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        retrieved_docs = []
        if results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                retrieved_docs.append({
                    'content': doc,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else 0
                })
        
        return retrieved_docs
    
    def generate_response(self, query: str, retrieved_docs: List[Dict]) -> str:
        """
        Retrieval edilen dokümanları kullanarak yanıt üretir.
        
        Args:
            query (str): Kullanıcı sorgusu
            retrieved_docs (List[Dict]): Retrieval edilen dokümanlar
            
        Returns:
            str: Üretilen yanıt
        """
        if not hasattr(self, 'model') or self.model is None:
            return "Gemini API konfigüre edilmemiş veya model yüklenemedi. Lütfen GEMINI_API_KEY'i kontrol edin ve geçerli bir API key kullandığınızdan emin olun."
        
        # Context oluştur
        context = "\n\n".join([doc['content'] for doc in retrieved_docs])
        
        # Prompt hazırla
        prompt = f"""
        Aşağıdaki bağlam bilgilerini kullanarak soruyu yanıtla. Yanıtın Türkçe olsun ve bağlam bilgilerine dayalı olsun.
        
        Bağlam:
        {context}
        
        Soru: {query}
        
        Yanıt:
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Yanıt üretme hatası: {e}")
            return f"Yanıt üretilirken hata oluştu: {e}"
    
    def chat(self, query: str) -> Dict[str, Any]:
        """
        Tam RAG pipeline'ını çalıştırır.
        
        Args:
            query (str): Kullanıcı sorgusu
            
        Returns:
            Dict[str, Any]: Yanıt ve metadata
        """
        # 1. Doküman retrieval
        retrieved_docs = self.retrieve_documents(query)
        
        # 2. Yanıt üretme
        response = self.generate_response(query, retrieved_docs)
        
        return {
            'query': query,
            'response': response,
            'retrieved_documents': retrieved_docs,
            'num_retrieved': len(retrieved_docs)
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
