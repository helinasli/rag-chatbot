"""
RAG (Retrieval-Augmented Generation) Sistemi Demo
Bu script, RAG sisteminin nasıl çalıştığını adım adım gösterir.
"""

import sys
import os
import time
from datetime import datetime

# Src klasörünü path'e ekle
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from rag_system import RAGSystem
except ImportError as e:
    print(f"❌ RAG sistemi yüklenemedi: {e}")
    print("Lütfen önce 'pip install -r requirements.txt' komutunu çalıştırın.")
    sys.exit(1)

def print_header(title):
    """Başlık yazdırma fonksiyonu"""
    print("\n" + "="*80)
    print(f"🔹 {title}")
    print("="*80)

def print_step(step_num, description):
    """Adım yazdırma fonksiyonu"""
    print(f"\n{step_num}️⃣ {description}")
    print("-" * 60)

def create_sample_data():
    """Örnek veri seti oluşturur"""
    return {
        "AI_Basics": "Yapay zeka (AI), makinelerin insan benzeri düşünme, öğrenme ve problem çözme yeteneklerini simüle etmesidir. AI sistemleri, veri analizi, pattern recognition ve karar verme süreçlerinde kullanılır.",
        
        "Machine_Learning": "Makine öğrenmesi, yapay zekanın bir alt dalıdır ve algoritmaların deneyimlerden öğrenmesini sağlar. Supervised, unsupervised ve reinforcement learning olmak üzere üç ana kategoriye ayrılır.",
        
        "Deep_Learning": "Derin öğrenme, çok katmanlı sinir ağları kullanarak karmaşık desenleri öğrenir. Görüntü işleme, doğal dil işleme ve ses tanıma gibi alanlarda devrim yaratmıştır.",
        
        "NLP": "Doğal dil işleme (NLP), bilgisayarların insan dilini anlaması, işlemesi ve üretmesini sağlayan AI dalıdır. Metin analizi, çeviri, sentiment analizi gibi uygulamalarda kullanılır.",
        
        "RAG_System": "Retrieval-Augmented Generation (RAG), bilgi retrieval ve text generation'ı birleştiren bir yaklaşımdır. Büyük dil modellerinin bilgi tabanından faydalanarak daha doğru yanıtlar üretmesini sağlar.",
        
        "Vector_DB": "Vektör veritabanları, yüksek boyutlu vektörleri verimli şekilde saklar ve benzerlik araması yapar. ChromaDB, Pinecone ve FAISS gibi çözümler popülerdir.",
        
        "Transformers": "Transformer, attention mechanism kullanan ve modern NLP modellerinin temelini oluşturan sinir ağı mimarisidir. BERT, GPT ve T5 gibi modeller bu mimariye dayanır.",
        
        "Embeddings": "Embedding modelleri, metinleri sayısal vektörlere dönüştürür. Bu vektörler, semantic similarity hesaplama ve information retrieval için kullanılır."
    }

def test_retrieval(rag_system, test_queries):
    """Doküman retrieval testleri"""
    print_step(4, "Doküman Retrieval Testleri")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Test {i}: {query}")
        
        start_time = time.time()
        retrieved_docs = rag_system.retrieve_documents(query, n_results=2)
        retrieval_time = time.time() - start_time
        
        print(f"⏱️  Retrieval süresi: {retrieval_time:.3f} saniye")
        print("📄 Bulunan dokümanlar:")
        
        for j, doc in enumerate(retrieved_docs, 1):
            similarity = 1 - doc['distance']
            print(f"   {j}. Benzerlik: {similarity:.3f}")
            print(f"      İçerik: {doc['content'][:80]}...")
        
        print("-" * 40)

def test_generation(rag_system):
    """Yanıt üretme testi"""
    print_step(5, "Yanıt Üretme Testi")
    
    test_query = "Yapay zeka ve makine öğrenmesi arasındaki fark nedir?"
    print(f"❓ Test sorusu: {test_query}\n")
    
    # Doküman retrieval
    print("🔍 Doküman retrieval yapılıyor...")
    retrieved_docs = rag_system.retrieve_documents(test_query, n_results=3)
    
    print("📄 Bulunan dokümanlar:")
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"   {i}. {doc['content'][:60]}...")
    
    # Yanıt üretme
    print("\n🧠 Yanıt üretiliyor...")
    response = rag_system.generate_response(test_query, retrieved_docs)
    print(f"\n🤖 Bot yanıtı:\n{response}")

def test_full_pipeline(rag_system):
    """Tam RAG pipeline testi"""
    print_step(6, "Tam RAG Pipeline Testleri")
    
    chat_queries = [
        "RAG sistemi nedir ve nasıl çalışır?",
        "Vektör veritabanları ne işe yarar?",
        "Embedding modelleri nasıl çalışır?",
        "Derin öğrenme hangi alanlarda kullanılır?"
    ]
    
    for i, query in enumerate(chat_queries, 1):
        print(f"\n💬 Sohbet {i}:")
        print(f"👤 Kullanıcı: {query}")
        
        start_time = time.time()
        result = rag_system.chat(query)
        total_time = time.time() - start_time
        
        print(f"🤖 Bot: {result['response']}")
        print(f"📊 Kullanılan doküman sayısı: {result['num_retrieved']}")
        print(f"⏱️  Toplam süre: {total_time:.2f} saniye")
        print("=" * 80)

def performance_evaluation(rag_system):
    """Performans değerlendirmesi"""
    print_step(7, "Performans Değerlendirmesi")
    
    # Sistem istatistikleri
    doc_count = rag_system.collection.count() if rag_system.collection else 0
    embedding_dim = rag_system.embedding_model.get_sentence_embedding_dimension()
    
    print(f"📚 Toplam doküman sayısı: {doc_count}")
    print(f"🔧 Embedding boyutu: {embedding_dim}")
    print(f"💾 Vektör veritabanı: ChromaDB")
    print(f"🧠 Generative model: Google Gemini Pro")
    
    # Retrieval performans testi
    sample_query = "Yapay zeka nedir?"
    
    print(f"\n⚡ Performans testi: '{sample_query}'")
    
    start_time = time.time()
    retrieved_docs = rag_system.retrieve_documents(sample_query, n_results=3)
    retrieval_time = time.time() - start_time
    
    print(f"🔍 Retrieval süresi: {retrieval_time:.3f} saniye")
    print(f"📄 Bulunan doküman sayısı: {len(retrieved_docs)}")
    
    if retrieved_docs:
        similarities = [1 - doc['distance'] for doc in retrieved_docs]
        avg_similarity = sum(similarities) / len(similarities)
        print(f"📊 Ortalama benzerlik skoru: {avg_similarity:.3f}")
        print(f"📈 En yüksek benzerlik: {max(similarities):.3f}")
        print(f"📉 En düşük benzerlik: {min(similarities):.3f}")

def main():
    """Ana demo fonksiyonu"""
    print_header("RAG (Retrieval-Augmented Generation) Sistemi Demo")
    print(f"📅 Demo başlangıç zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 1. RAG Sistemi Kurulumu
        print_step(1, "RAG Sistemi Kurulumu")
        print("🚀 RAG sistemi başlatılıyor...")
        rag = RAGSystem(embedding_model_name="all-MiniLM-L6-v2")
        print("✅ RAG sistemi başarıyla kuruldu!")
        
        print(f"\n📊 Sistem bilgileri:")
        print(f"   - Embedding model: all-MiniLM-L6-v2")
        print(f"   - Vektör DB: ChromaDB")
        print(f"   - LLM: Google Gemini Pro")
        
        # 2. Veri Yükleme ve İşleme
        print_step(2, "Veri Yükleme ve İşleme")
        sample_data = create_sample_data()
        
        print("📚 Örnek veri seti yükleniyor...")
        rag.load_and_process_data(data_dict=sample_data)
        print(f"✅ {len(sample_data)} doküman başarıyla yüklendi ve vektörize edildi!")
        
        print("\n📋 Yüklenen dokümanlar:")
        for i, (key, value) in enumerate(sample_data.items(), 1):
            print(f"   {i}. {key}: {value[:50]}...")
        
        # 3. API Key Kontrolü
        print_step(3, "API Key Kontrolü")
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            print("✅ Gemini API key bulundu!")
        else:
            print("⚠️  Gemini API key bulunamadı.")
            print("💡 .env dosyasında GEMINI_API_KEY'i ayarlayın veya")
            print("   export GEMINI_API_KEY='your_key_here' komutunu çalıştırın.")
        
        # Test sorguları
        test_queries = [
            "Yapay zeka nedir?",
            "Makine öğrenmesi türleri neler?",
            "RAG sistemi nasıl çalışır?",
            "Transformer mimarisi nedir?"
        ]
        
        # 4. Doküman Retrieval Testleri
        test_retrieval(rag, test_queries)
        
        # 5. Yanıt Üretme Testi (sadece API key varsa)
        if api_key:
            test_generation(rag)
            
            # 6. Tam RAG Pipeline
            test_full_pipeline(rag)
        else:
            print("\n⚠️  API key olmadığı için generation testleri atlanıyor.")
        
        # 7. Performans Değerlendirmesi
        performance_evaluation(rag)
        
        # Sonuç
        print_header("Demo Tamamlandı")
        print("✅ RAG sistemi başarıyla test edildi!")
        print("\n🚀 Sonraki adımlar:")
        print("   1. Web arayüzünü çalıştırın: streamlit run app.py")
        print("   2. Kendi veri setinizi yükleyin")
        print("   3. API key'inizi ayarlayın (eğer henüz yapmadıysanız)")
        print("   4. Farklı sorular deneyin")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo kullanıcı tarafından durduruldu.")
    except Exception as e:
        print(f"\n❌ Demo sırasında hata oluştu: {e}")
        print("💡 Lütfen requirements.txt dosyasındaki bağımlılıkları kontrol edin.")
    finally:
        print(f"\n📅 Demo bitiş zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
