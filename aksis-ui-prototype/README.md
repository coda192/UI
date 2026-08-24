# ⚡ AKSIS - Kurumsal Makine Öğrenmesi & MLOps Platformu

Bu proje, **AKSIS Machine Learning Framework** için geliştirilmiş kurumsal kullanıcı arayüzü ve API sözleşme katmanıdır. 

Proje, **Sunum Katmanı (Streamlit)** ile **ML Motoru (FastAPI & AKSIS)** mimarilerini birbirinden tamamen izole eden (decoupled) modern bir **REST API ve Adapter Deseni** üzerine inşa edilmiştir.

---

## 🏛️ Mimari Yapı

```mermaid
graph TD
    User([Kullanıcı / Tarayıcı]) <--> Streamlit[Frontend: Streamlit - app.py]
    Streamlit <-->|HTTP REST API| Client[frontend/api/client.py: AksisClient]
    Client <-->|JSON İstek / Yanıt| FastAPI[Backend: FastAPI - main.py]
    FastAPI --> Deps[api/deps.py: get_service]
    Deps --> Factory[services/factory.py: get_aksis_service]
    
    Factory -->|AKSIS_PROVIDER=mock| MockService[services/mock_service.py: MockAksisService]
    Factory -->|AKSIS_PROVIDER=aksis| RealService[services/aksis_service.py: RealAksisService]
    
    MockService --> DemoData[backend/demo/ (Fixtures & Scenarios)]
    RealService -.->|AKSIS_INTEGRATION_POINT| RealAKSIS[Kurumsal AKSIS ML Motoru]
```

### Temel Bileşenler:
1. **Frontend (Streamlit):** `frontend/` dizinindedir. `st.navigation` çoklu sayfa mimarisini kullanır. Yalnızca sunumdan sorumludur; doğrudan ML kodu çalıştırmaz, backend ile HTTP üzerinden haberleşir.
2. **Backend (FastAPI):** `backend/` dizinindedir. Kararlı REST API (`/api/v1`) endpoint'lerini ve Pydantic tip doğrulama şemalarını sunar.
3. **Servis Katmanı (Factory & Adapter):**
   - **Mock Modu (`MockAksisService`):** Yerel geliştirme ve demo için in-memory çalışan, threading ile asenkron eğitimi taklit eden ve interaktif Plotly HTML grafikleri üreten simülasyon modudur.
   - **AKSIS Modu (`RealAksisService`):** Şirket bilgisayarındaki gerçek kurumsal AKSIS ML kütüphanesine, registry'lerine ve `outputs/` klasörüne bağlanacak adaptör katmanıdır.

---

## 🚀 Kurulum ve Çalıştırma

### 1. Ortam ve Bağımlılıkların Hazırlanması

```powershell
# 1. Sanal ortam oluştur ve aktif et
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Proje dizinine gir ve kütüphaneleri kur
cd aksis-ui-prototype
python -m pip install -r requirements.txt

# 3. .env konfigürasyon dosyasını oluştur
copy .env.example .env
```
*(Mock modda yerel test için `.env` içinde `AKSIS_PROVIDER=mock` ayarlı olmalıdır).*

---

### 2. Uygulamayı Başlatma (2 Terminal)

#### 🟢 1. Terminal: FastAPI Backend
```powershell
python -m uvicorn backend.main:app --reload
```
* Swagger API Dokümantasyonu: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

#### 🟢 2. Terminal: Streamlit Frontend
```powershell
cd frontend
streamlit run app.py
```
* Kullanıcı Arayüzü: [http://localhost:8501](http://localhost:8501)

---

## 📄 Sayfa Yapısı ve Modüller

1. **🏠 Ana Sayfa (`0_home.py`):** Kurumsal vizyon, 4 temel değer kartı, interaktif mimari şeması (Mermaid & görsel desteği) ve 4 adımlı iş akışı.
2. **📊 Sistem Özeti (`1_dashboard.py`):** Kayıtlı model ve algoritma sayısı, görevler, model kataloğu ve her model için **Model Karar Destek Rehberi (Güçlü/Zayıf Yönler)**.
3. **📁 Veri Setleri (`2_datasets.py`):** Satır/sütun metrikleri, hedef değişken, uyumlu görevler ve sütun veri tipi / eksik değer analiz tablosu.
4. **⚙️ Deney Yapılandırma (`3_experiment_config.py`):** Akıllı görev/model filtreleme, anlık model karar destek kutusu, önişleme ve doğrulama parametreleri.
5. **📈 Çalıştırma & Sonuçlar (`4_execution_results.py`):** Asenkron çalıştırma, canlı polling ilerleme çubuğu, metrikler ve **interaktif Plotly HTML grafikleri**.
6. **🚀 Modeller & Toplu Tahmin (`5_artifacts_inference.py`):** Eğitilmiş model çıktıları ve yeni veri setleri üzerinde toplu çıkarım (Batch Inference).

---

## 🔗 Şirket Entegrasyonu

Şirket bilgisayarındaki gerçek AKSIS kütüphanesine bağlanma adımları için [INTEGRATION.md](INTEGRATION.md) dosyasını inceleyebilirsiniz.
