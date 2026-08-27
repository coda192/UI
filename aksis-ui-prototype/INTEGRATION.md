# AKSIS UI — Gerçek Çerçeve (Framework) Entegrasyon ve Hazırlık Kılavuzu

Bu belge, ofis/şirket bilgisayarındaki tescilli **AKSIS Makine Öğrenmesi Çerçevesi** ile mevcut UI/API mimarisi arasındaki entegrasyonu en az adaptör koduyla, var olan mimariyi bozmadan ve hiçbir tescilli/iç kaynak kodunu dışarı çıkarmaya gerek kalmadan gerçekleştirmek üzere hazırlanmıştır.

```text
Streamlit Configuration UI
        ↓ HTTP / JSON
FastAPI (Pydantic Validation)
        ↓
RealAksisService (Adapter)
        ↓
AKSIS ExperimentConfig
        ↓
run_experiment(exp)
```

---

## 1. INFORMATION REQUIRED FROM AKSIS (Önceliklendirilmiş Yapısal Bilgi Talebi)

> [!IMPORTANT]
> **Güvenlik Notu:** Fonksiyon gövdelerine, iş mantığı kodlarına veya kurumsal verilere ihtiyaç **yoktur**. Yalnızca `dataclass`/`class` alan adları, veri tipleri (`typing`), varsayılan değerler ve kabul edilen Literal/Enum seçenekleri yeterlidir.

### PRIORITY 1 — `ExperimentConfig` ve Ana Konfigürasyon Yapısı
* **Soru 1.1:** `ExperimentConfig` sınıfının/dataclass'ının tüm alan adları ve tipleri nelerdir?
* **Soru 1.2:** Alt konfigürasyonlar (Model, Preprocessing, Validation, Tuning vb.) iç içe (nested) nesneler olarak mı tanımlıdır, yoksa düz (flat) bir yapıda mıdır?
* **Soru 1.3:** `ExperimentConfig` sınıfının bir sözlükten (`dict` / `json`) doğrudan ayağa kalkmasını sağlayan `from_dict(...)` veya `**dict` desteği var mıdır?

---

### PRIORITY 2 — Model ve Algoritma Konfigürasyonu (`ModelConfig`)
* **Soru 2.1:** `ModelConfig` alan adları ve tipleri nelerdir? (Örn: `algorithm: str`, `preset: Optional[str]`, `overrides: dict`)
* **Soru 2.2:** AKSIS bünyesinde mevcut bir model/algoritma registry'si var mıdır? (Örn: `model_registry.get_supported(task)` gibi çalışan bir fonksiyon/sözlük var mı?)
* **Soru 2.3:** Görev bazlı (Sınıflandırma, Regresyon, Anomali Tespiti) uyumlu model listesi kodda nasıl tutulmaktadır?

---

### PRIORITY 3 — Hiperparametre Optimizasyonu (`TuningConfig`)
* **Soru 3.1:** `TuningConfig` alan adları ve tipleri nelerdir? (Örn: `enabled: bool`, `sampler: str`, `pruner: str`, `n_trials: int`, `scoring: str`, `aggregation: str` vb.)
* **Soru 3.2:** Desteklenen Sampler (TPE, Random vb.), Pruner (Hyperband, Median vb.) ve Metrik (F1, RMSE vb.) seçenekleri Literal/Enum olarak mı tanımlıdır?
* **Soru 3.3:** Optuna / Tuning kapalı olduğunda bu konfigürasyon `None` mı geçirilir, yoksa `enabled=False` şeklinde bir bayrak (flag) mı kullanılır?

---

### PRIORITY 4 — Önişleme Konfigürasyonu (`PreprocessConfig` / `CorePreProcesser`)
* **Soru 4.1:** Önişleme adımlarını (Missing value, Encoding, Scaling) kontrol eden alan adları nelerdir?
* **Soru 4.2:** Bir önişleme adımı devre dışı bırakıldığında `None` mı atanır, yoksa `enabled: bool` veya `strategy="passthrough"` gibi bir değer mi alır?
* **Soru 4.3:** Sayısal ve kategorik sütunlar için ayrı ayrı strateji tanımlanabiliyor mu?

---

### PRIORITY 5 — Veri Seti Sözleşmesi (`DatasetSpec` / `DataSpec`)
* **Soru 5.1:** Bir veri setini sorgulamak için kullanılan anahtar alan nedir? (`dataset_id`, `name`, `path` vb.)
* **Soru 5.2:** `DataSpec` veya `DatasetSpec` nesnesi dışarıya hangi meta verileri açar? (Örn: `target: str`, `id_columns: list`, `row_count: int`, `feature_names: list`)
* **Soru 5.3:** Veri setlerini listeleyen merkezi bir katalog/registry fonksiyonu var mıdır? (Örn: `dataset_catalog.list()`)

---

### PRIORITY 6 — Deney Koşucusu (`run_experiment`)
* **Soru 6.1:** Deneyi başlatan ana fonksiyonun imzası nedir? (`run_experiment(exp: ExperimentConfig)` mi?)
* **Soru 6.2:** Fonksiyon çalışma biçimi senkron mudur (bloklar mı)?
* **Soru 6.3:** Fonksiyon ne döndürür? (`None`, `ResultObject`, `dict` vb.)
* **Soru 6.4:** Hata durumlarında fırlatılan özel bir Exception sınıfı var mıdır?

---

### PRIORITY 7 — Deney Çıktıları ve Dizin Yapısı (`outputs/`)
* **Soru 7.1:** Deney tamamlandığında metrikler nereye yazılır? (Bellekte mi döner, `outputs/{exp_id}/metrics.json` gibi bir dosyaya mı yazılır?)
* **Soru 7.2:** Üretilen Plotly `.html` grafikleri hangi klasör altında ve hangi isimlendirme kuralıyla saklanır?
* **Soru 7.3:** Eğitilmiş model dosyası (artifact) nereye ve hangi uzantıyla kaydedilir? (`model.joblib`, `model.cbm`, `model.pkl` vb.)

---

### PRIORITY 8 — Toplu Çıkarım / Tahmin (`Inference`)
* **Soru 8.1:** Eğitilmiş bir model ile yeni veri üzerinde tahmin üreten fonksiyonun adı ve imzası nedir?
* **Soru 8.2:** Giriş olarak `model_path` ve `dataset_id` mi alır, yoksa `DataSpec` nesnesi mi bekler?
* **Soru 8.3:** Çıktı olarak pandas DataFrame mi döner, numpy array mi, yoksa bir dosyaya mı yazar?

---

### PRIORITY 9 — Loglama Yapısı
* **Soru 9.1:** Standart Python `logging` mi kullanılmaktadır?
* **Soru 9.2:** Deney bazında izole bir log dosyası (`outputs/{exp_id}/run.log`) üretiliyor mu?

---

## 2. CONFIG MAPPING TEMPLATE (İzlenebilirlik Tablosu)

Aşağıdaki tablo, Streamlit kullanıcı arayüzündeki her bir alanın, FastAPI JSON yüküne ve oradan gerçek AKSIS `ExperimentConfig` nesnesine nasıl eşleşeceğini gösterir:

| UI Alanı (Streamlit) | API JSON Alanı (`ExperimentCreateRequest`) | Beklenen AKSIS Konfigürasyon Alanı | Durum |
| :--- | :--- | :--- | :--- |
| **Veri Seti Seçimi** | `dataset_id` | `ExperimentConfig.dataset_id` | `NEEDS_AKSIS_CONFIRMATION` |
| **Deney Adı** | `name` | `ExperimentConfig.name` *(veya `exp_name`)* | `NEEDS_AKSIS_CONFIRMATION` |
| **Öğrenme Türü** | `learning_type` | `ExperimentConfig.learning_type` | `NEEDS_AKSIS_CONFIRMATION` |
| **Görev (Task)** | `task` | `ExperimentConfig.task` | `NEEDS_AKSIS_CONFIRMATION` |
| **Çalışma Modu** | `mode` | `ExperimentConfig.mode` *(örn: train, tune, eval)* | `NEEDS_AKSIS_CONFIRMATION` |
| **Algoritma Seçimi** | `model.algorithm` | `ExperimentConfig.model_config.algorithm` | `NEEDS_AKSIS_CONFIRMATION` |
| **Model Hazır Ayarı** | `model.preset` | `ExperimentConfig.model_config.preset` | `NEEDS_AKSIS_CONFIRMATION` |
| **Model Parametreleri** | `model.overrides` | `ExperimentConfig.model_config.params` | `NEEDS_AKSIS_CONFIRMATION` |
| **Eksik Değer Yöntemi** | `preprocessing.missing_value` | `ExperimentConfig.preprocess_config.imputation` | `NEEDS_AKSIS_CONFIRMATION` |
| **Kategorik Kodlama** | `preprocessing.encoding` | `ExperimentConfig.preprocess_config.encoding` | `NEEDS_AKSIS_CONFIRMATION` |
| **Ölçeklendirme** | `preprocessing.scaling` | `ExperimentConfig.preprocess_config.scaling` | `NEEDS_AKSIS_CONFIRMATION` |
| **Doğrulama Stratejisi** | `validation.strategy` | `ExperimentConfig.validation_config.strategy` | `NEEDS_AKSIS_CONFIRMATION` |
| **Test Kümesi Oranı** | `validation.test_size` | `ExperimentConfig.validation_config.test_size` | `NEEDS_AKSIS_CONFIRMATION` |
| **Tuning Aktif/Pasif** | `tuning.enabled` | `ExperimentConfig.tuning_config.enabled` | `NEEDS_AKSIS_CONFIRMATION` |
| **Optuna Sampler** | `tuning.sampler` | `ExperimentConfig.tuning_config.sampler` | `NEEDS_AKSIS_CONFIRMATION` |
| **Optuna Pruner** | `tuning.pruner` | `ExperimentConfig.tuning_config.pruner` | `NEEDS_AKSIS_CONFIRMATION` |
| **Deneme Sayısı (Trials)**| `tuning.trials` | `ExperimentConfig.tuning_config.n_trials` | `NEEDS_AKSIS_CONFIRMATION` |
| **Optimizasyon Skoru** | `tuning.scoring` | `ExperimentConfig.tuning_config.metric` | `NEEDS_AKSIS_CONFIRMATION` |

---

## 3. MINIMAL INTEGRATION PLAN (En Küçük Adaptör Mimarisi)

Ekstra hiçbir çeviri katmanı (`TranslationEngine`, `ConfigBuilder` vb.) kurmadan, `RealAksisService` adaptör uygulaması:

```python
import os
import json
import time
import threading
from datetime import datetime
from typing import List
from backend.services.base import AksisService
from backend.schemas import (
    CapabilityResponse, DatasetMetadata, ExperimentCreateRequest,
    ExperimentMetadata, ExperimentResultResponse, VisualizationData,
    MetricsData, ArtifactMetadata, InferenceRequest, InferenceResponse
)

# 1. Gerçek AKSIS modüllerinin import edilmesi (Aynı proje içi)
from aksis.config import ExperimentConfig, ModelConfig, PreprocessConfig, TuningConfig, ValidationConfig
from aksis.runner import run_experiment
from aksis.registry import model_registry, task_registry
from aksis.data import dataset_catalog
from aksis.inference import predict_batch

class RealAksisService(AksisService):
    def __init__(self):
        self._in_memory_status = {}  # exp_id -> status ('configured', 'running', 'completed', 'failed')

    def get_capabilities(self) -> CapabilityResponse:
        # Doğrudan mevcut registry'lerden dinamik oku
        return CapabilityResponse(
            learning_types=["supervised", "unsupervised"],
            tasks=task_registry.get_all(),
            algorithms=model_registry.get_all_by_task(),
            model_presets=["fast", "accurate", "interpretable"],
            preprocessing_strategies=PreprocessConfig.get_supported_strategies(),
            validation_options=["holdout", "kfold", "stratified_kfold"],
            tuning_options=TuningConfig.get_supported_options()
        )

    def list_datasets(self) -> List[DatasetMetadata]:
        # Doğrudan dataset katalogundan meta verileri çek
        specs = dataset_catalog.list_specs()
        return [
            DatasetMetadata(
                id=s.id,
                name=s.name,
                row_count=s.row_count,
                column_count=s.column_count,
                columns=s.get_column_metadata(),
                target=s.target,
                compatible_tasks=s.compatible_tasks
            ) for s in specs
        ]

    def create_experiment(self, req: ExperimentCreateRequest) -> ExperimentMetadata:
        exp_id = f"exp_{req.name}_{int(time.time())}"
        self._in_memory_status[exp_id] = {"status": "configured", "request": req}
        return ExperimentMetadata(id=exp_id, name=req.name, status="configured", created_at=datetime.now())

    def run_experiment(self, experiment_id: str) -> None:
        req = self._in_memory_status[experiment_id]["request"]
        
        # 2. JSON Sözlüğünden Doğrudan AKSIS ExperimentConfig Nesnesini Oluştur
        # (Eğer AKSIS from_dict destekliyorsa: exp_config = ExperimentConfig.from_dict(req.model_dump()))
        exp_config = ExperimentConfig(
            experiment_id=experiment_id,
            dataset_id=req.dataset_id,
            task=req.task,
            mode=req.mode,
            model_config=ModelConfig(**req.model.model_dump()),
            preprocess_config=PreprocessConfig(**req.preprocessing.model_dump()) if req.preprocessing else None,
            tuning_config=TuningConfig(**req.tuning.model_dump()) if req.tuning and req.tuning.enabled else None,
            validation_config=ValidationConfig(**req.validation.model_dump()) if req.validation else None
        )
        
        self._in_memory_status[experiment_id]["status"] = "running"

        # 3. Asenkron (Non-blocking) Olarak AKSIS Çalıştırıcısını Tetikle
        def _execute():
            try:
                run_experiment(exp_config)
                self._in_memory_status[experiment_id]["status"] = "completed"
            except Exception as e:
                self._in_memory_status[experiment_id]["status"] = "failed"
                self._in_memory_status[experiment_id]["error"] = str(e)

        threading.Thread(target=_execute, daemon=True).start()

    def get_experiment_results(self, experiment_id: str) -> ExperimentResultResponse:
        output_dir = os.path.join("outputs", experiment_id)
        
        # 4. Diskteki outputs/{exp_id}/ Klasörünü Tara
        with open(os.path.join(output_dir, "metrics.json"), "r", encoding="utf-8") as f:
            metrics_dict = json.load(f)
            
        visualizations = []
        for fname in os.listdir(output_dir):
            if fname.endswith(".html"):
                with open(os.path.join(output_dir, fname), "r", encoding="utf-8") as f:
                    visualizations.append(VisualizationData(
                        type=fname.replace(".html", ""),
                        title=fname.replace(".html", "").replace("_", " ").title(),
                        html_content=f.read()
                    ))
                    
        return ExperimentResultResponse(
            experiment_id=experiment_id,
            task=self._in_memory_status[experiment_id]["request"].task,
            status=self._in_memory_status[experiment_id]["status"],
            algorithm=self._in_memory_status[experiment_id]["request"].model.algorithm,
            has_ground_truth=True,
            metrics=MetricsData(**metrics_dict),
            visualizations=visualizations
        )
```

---

## 4. AKSIS-SIDE CHANGE ASSESSMENT (AKSIS Tarafı Değişiklik Değerlendirmesi)

| Kategori | AKSIS Tarafı Durumu | Gerekçe / Açıklama |
| :--- | :--- | :--- |
| **A — Değişiklik Gerekmez (No Change)** | `run_experiment(exp_config)` | Mevcut koşucu doğrudan çağrılabilir. |
| **A — Değişiklik Gerekmez (No Change)** | `DataAnalyzer`, `CorePreProcesser` | Pipeline iç işleyişine API asla müdahale etmez. |
| **A — Değişiklik Gerekmez (No Change)** | Model & Pipeline Çalıştırıcıları | Eğitim, tuning ve değerlendirme AKSIS içinde kalır. |
| **B — Küçük Erişim Yardımcısı (Optional)** | `registry.get_all()` / `catalog.list()` | Eğer registry'ler sözlük olarak dışa açık değilse küçük bir public getter eklenebilir. |
| **C — Çıktı Standardizasyonu (Optional)** | `outputs/{exp_id}/metrics.json` | Eğer metrikler diske yazılmıyorsa, koşucu sonunda metriklerin bir JSON dosyasına kaydedilmesi sağlanabilir. |
| **D — Reddedilecek Değişiklikler (Reject)** | Streamlit / HTTP / FastAPI kodları | AKSIS çekirdeğine asla UI veya Web kütüphanesi bağımlılığı eklenmeyecektir. |

---

## 5. COMPANY-PC INTEGRATION CHECKLIST (Şirket Bilgisayarı Kontrol Listesi)

Şirket bilgisayarına geçtiğinizde aşağıdaki adımları sırayla inceleyip yalnızca yapısal alan adlarını not etmeniz yeterlidir:

```text
[ ] 1. ExperimentConfig Sınıfı
       - Dosya yolu nedir?
       - Hangi alanları (fields) ve tipleri alır?
       - from_dict() veya dict unpack (**kwargs) destekliyor mu?

[ ] 2. ModelConfig / Model Tanımı
       - Model konfigürasyonu hangi alanları içerir? (algorithm, preset, params vb.)
       - Desteklenen modeller listesi hangi registry'den okunabilir?

[ ] 3. PreprocessConfig / Önişleme Ayarları
       - Missing value, scaling ve encoding için kullanılan alan adları nelerdir?
       - Pasif yapmak için None mı verilir, flag mi kullanılır?

[ ] 4. TuningConfig / Hiperparametre Ayarları
       - Sampler, pruner, n_trials ve scoring alan adları nelerdir?
       - Tuning kapalıyken alan nasıl set edilir?

[ ] 5. ValidationConfig / Doğrulama Ayarları
       - K-Fold, Holdout ve test_size parametreleri nasıl isimlendirilmiştir?

[ ] 6. DatasetSpec / Veri Kataloğu
       - Kayıtlı veri setlerinin listesi ve meta verileri (target, id_cols) nereden çekilir?

[ ] 7. run_experiment Giriş Noktası
       - Çağrılan ana fonksiyonun adı, import yolu ve imzası nedir?
       - Senkron mu çalışır, asenkron mu?

[ ] 8. outputs/ Çıktı Klasörü ve Dosya Formatları
       - Metrikler hangi dosyaya kaydedilir? (metrics.json vb.)
       - Plotly grafikleri hangi isimlerle .html olarak kaydedilir?
       - Model artifact dosyası nerede depolanır?

[ ] 9. Inference / Tahmin Fonksiyonu
       - Kaydedilen model ile yeni veri seti üzerinde tahmin çalıştıran fonksiyonun adı ve parametreleri nelerdir?
```
