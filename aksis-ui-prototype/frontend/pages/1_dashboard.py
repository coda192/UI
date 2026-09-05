import streamlit as st
import pandas as pd
from api.client import AksisAPIError
from utils.model_guidance import get_algorithm_metadata, get_algorithm_display_name

st.title("📊 Sistem Özeti & Registry Kataloğu")
st.caption("AKSIS bünyesinde kayıtlı makine öğrenmesi algoritmaları, veri setleri, önişleme yetenekleri ve deney geçmişi.")

client = st.session_state.client

# Sunum Etiketleri (Presentation Aliases)
TASK_NAMES_TR = {
    "classification": "Sınıflandırma (Classification)",
    "regression": "Regresyon (Regression)",
    "anomaly_detection": "Anomali Tespiti (Anomaly Detection)"
}

LEARNING_TYPES_TR = {
    "supervised": "Denetimli (Supervised)",
    "unsupervised": "Denetimsiz (Unsupervised)"
}

MODE_NAMES_TR = {
    "train": "Model Eğitimi (train)",
    "tune": "Hiperparametre Optimizasyonu (tune)",
    "predict": "Toplu Çıkarım / Tahmin (predict)"
}

PRESET_NAMES_TR = {
    "baseline": "Temel Seviye (baseline)",
    "fast": "Hızlı (fast)",
    "strong": "Yüksek Başarım (strong)",
    "custom": "Özel Ayar (custom)",
    "default": "Varsayılan (default)"
}

STATUS_TR = {
    "configured": "⚙️ Yapılandırıldı",
    "running": "⏳ Eğitiliyor...",
    "completed": "✅ Tamamlandı",
    "failed": "❌ Başarısız"
}

# 1. API'den Dinamik Yetenek Verilerini Çek
capabilities = None
try:
    capabilities = client.get_capabilities()
except (AksisAPIError, Exception) as e:
    st.error(f"⚠️ Sistem yetenekleri (capabilities) backend servisinden yüklenemedi: {str(e)}")
    st.info("Lütfen backend API servisinin (`http://127.0.0.1:8000`) çalıştığından ve erişilebilir olduğundan emin olun.")
    st.stop()

if not capabilities:
    st.error("⚠️ Backend'den boş yetenek yanıtı alındı.")
    st.stop()

# Veri setleri ve deney geçmişini sorgula
try:
    datasets = client.get_datasets()
except Exception:
    datasets = []

try:
    experiments = client.list_experiments()
except Exception:
    experiments = []

# Algoritma ve Görev Sayılarını Dinamik Olarak Hesapla
algorithms_dict = capabilities.get("algorithms", {})
cls_algos = algorithms_dict.get("classification", [])
reg_algos = algorithms_dict.get("regression", [])
anom_algos = algorithms_dict.get("anomaly_detection", [])

total_algos = sum(len(v) for v in algorithms_dict.values())
supported_modes = capabilities.get("modes", [])
learning_types = capabilities.get("learning_types", [])
tasks_dict = capabilities.get("tasks", {})
total_tasks = sum(len(v) for v in tasks_dict.values())

# 1. ÜST YÖNETİCİ METRİK KARTLARI (Tümü API'den Dinamik Hesaplanır)
m1, m2, m3, m4 = st.columns(4)
with m1:
    with st.container(border=True):
        st.metric("🧠 Kayıtlı Model & Algoritma", f"{total_algos} Adet")
with m2:
    with st.container(border=True):
        st.metric("🎯 Desteklenen ML Görevi", f"{total_tasks} Görev")
with m3:
    with st.container(border=True):
        st.metric("📁 Kayıtlı Veri Seti", f"{len(datasets)} Veri Seti")
with m4:
    with st.container(border=True):
        st.metric("🧪 Kayıtlı Deney", f"{len(experiments)} Deney")

st.divider()

# 2. GÖREV VE ÇALIŞMA MODU DAĞILIMI (Dinamik Framework Yetenekleri)
st.subheader("⚙️ Çerçeve Yetenekleri ve Modları (Framework Capabilities)")

col_cap1, col_cap2, col_cap3 = st.columns(3)

with col_cap1:
    with st.container(border=True):
        st.markdown("#### 🎯 Görev Dağılımı")
        st.write(f"- **Sınıflandırma:** `{len(cls_algos)}` algoritma")
        st.write(f"- **Regresyon:** `{len(reg_algos)}` algoritma")
        st.write(f"- **Anomali Tespiti:** `{len(anom_algos)}` algoritma")

with col_cap2:
    with st.container(border=True):
        st.markdown("#### 🚀 Desteklenen Deney Modları")
        for mode in supported_modes:
            st.write(f"- `{mode}` ({MODE_NAMES_TR.get(mode, mode)})")

with col_cap3:
    with st.container(border=True):
        st.markdown("#### 🧠 Öğrenme Türleri")
        for lt in learning_types:
            st.write(f"- `{lt}` ({LEARNING_TYPES_TR.get(lt, lt)})")

st.divider()

# 3. MODEL & ALGORİTMA KATALOĞU (Registry & Karar Destek Rehberi)
st.subheader("📚 Kayıtlı Model & Algoritma Kataloğu (Registry & Rehber)")
st.caption("Her algoritmanın güçlü/zayıf yönleri ve kurumsal karar destek için önerilen kullanım senaryoları.")

task_keys = list(algorithms_dict.keys())
if task_keys:
    tab_names = [f"🎯 {TASK_NAMES_TR.get(k, k.title())} ({len(algorithms_dict.get(k, []))})" for k in task_keys]
    tabs = st.tabs(tab_names)
    
    for idx, task_key in enumerate(task_keys):
        algos = algorithms_dict.get(task_key, [])
        with tabs[idx]:
            st.markdown(f"**Bu görev için API üzerinden sağlanan {len(algos)} algoritma:**")
            
            # Algoritmaları 2 sütunlu kartlar halinde listele
            cols = st.columns(2)
            for i, algo in enumerate(algos):
                algo_meta = get_algorithm_metadata(algo, capabilities)
                display_title = algo_meta.get("display_name") or algo
                description = algo_meta.get("description", "Model açıklaması sağlanmamış.")
                strengths = algo_meta.get("strengths", [])
                limitations = algo_meta.get("limitations", [])
                best_for = algo_meta.get("best_for", [])
                
                with cols[i % 2]:
                    with st.container(border=True):
                        st.markdown(f"### ⚡ `{algo}`")
                        st.markdown(f"**{display_title}**")
                        st.write(description)
                        if best_for:
                            st.markdown(f"**🎯 En Uygun Senaryo:** {', '.join(best_for) if isinstance(best_for, list) else best_for}")
                        with st.expander("🔍 Güçlü ve Dikkat Edilmesi Gereken Yönler", expanded=False):
                            if strengths:
                                st.markdown("**✅ Güçlü Yönler:**")
                                for s in strengths:
                                    st.write(f"- {s}")
                            if limitations:
                                st.markdown("**⚠️ Dikkat Edilmesi Gerekenler:**")
                                for lim in limitations:
                                    st.write(f"- {lim}")
else:
    st.info("Kayıtlı algoritma bulunamadı.")

st.divider()

# 4. PREPROCESSING, PRESETS & TUNING YETENEKLERİ
st.subheader("🔧 Önişleme, Preset ve Optimizasyon Yetenekleri")

p_col1, p_col2, p_col3 = st.columns(3)

prep_strats = capabilities.get("preprocessing_strategies", {})
presets = capabilities.get("model_presets", [])
tuning_opts = capabilities.get("tuning_options", {})
val_opts = capabilities.get("validation_options", [])

with p_col1:
    with st.container(border=True):
        st.markdown("#### 🩹 Önişleme Stratejileri")
        st.caption("Eksik Değerler (Imputation):")
        for s in prep_strats.get("missing_value", []):
            st.write(f"- `{s}`")
        st.caption("Kategorik Kodlama (Encoding):")
        for s in prep_strats.get("encoding", []):
            st.write(f"- `{s}`")
        st.caption("Ölçeklendirme (Scaling):")
        for s in prep_strats.get("scaling", []):
            st.write(f"- `{s}`")

with p_col2:
    with st.container(border=True):
        st.markdown("#### 🎛️ Model Hazır Ayarları (Presets)")
        for pr in presets:
            st.write(f"- `{pr}`: {PRESET_NAMES_TR.get(pr, pr)}")
        st.markdown("#### 📐 Doğrulama Seçenekleri")
        for v in val_opts:
            st.write(f"- `{v}`")

with p_col3:
    with st.container(border=True):
        st.markdown("#### ⚡ Optuna Tuning Yetenekleri")
        st.caption("Samplers:")
        for s in tuning_opts.get("sampler", []):
            st.write(f"- `{s}`")
        st.caption("Pruners:")
        for p in tuning_opts.get("pruner", []):
            st.write(f"- `{p}`")
        if "space_preset" in tuning_opts:
            st.caption("Space Presets:")
            for sp in tuning_opts.get("space_preset", []):
                st.write(f"- `{sp}`")

st.divider()

# 5. SON DENEYLER TABLOSU
st.subheader("🕒 Son Deneyler & İzlenebilirlik")
if experiments:
    df_exp = pd.DataFrame(experiments)
    if "status" in df_exp.columns:
        df_exp["status"] = df_exp["status"].map(lambda s: STATUS_TR.get(s, s))
    df_exp = df_exp.rename(columns={
        "id": "Deney ID",
        "name": "Deney Adı",
        "status": "Güncel Durum",
        "created_at": "Oluşturulma Tarihi"
    })
    st.dataframe(df_exp, use_container_width=True)
else:
    st.info("💡 Henüz kayıtlı bir deney bulunmamaktadır. 'Deney Yapılandırma' sayfasından yeni bir deney oluşturabilirsiniz.")
