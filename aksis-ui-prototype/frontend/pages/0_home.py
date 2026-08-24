import streamlit as st

# Hero Banner & Başlık
st.markdown("""
<div style="background: linear-gradient(135deg, #1E3A8A 0%, #0F172A 100%); padding: 28px; border-radius: 14px; margin-bottom: 24px; color: white; box-shadow: 0 4px 14px rgba(0,0,0,0.15);">
    <div style="display: flex; align-items: center; justify-content: space-between;">
        <div>
            <h1 style="margin: 0; font-size: 2.2rem; font-weight: 700; color: #FFFFFF; letter-spacing: -0.5px;">AKSIS</h1>
            <p style="margin: 6px 0 0 0; font-size: 1.1rem; color: #94A3B8; font-weight: 400;">Kurumsal Makine Öğrenmesi & MLOps Yönetim Platformu</p>
        </div>
        <div style="background: rgba(255,255,255,0.1); padding: 8px 16px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.3);">
            ● Kurumsal Sürüm v2.0
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# 4 Temel Kurumsal Değer Kartı
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    with st.container(border=True):
        st.markdown("### 🏛️ Merkezi Standart")
        st.caption("Kurum genelinde ortak teknik standartlar ve merkezi MLOps altyapısı.")

with kpi2:
    with st.container(border=True):
        st.markdown("### ⚡ Hızlı Geliştirme")
        st.caption("Otomatik ön işleme, hiperparametre optimizasyonu ve modüler pipeline.")

with kpi3:
    with st.container(border=True):
        st.markdown("### 🔍 Tam İzlenebilirlik")
        st.caption("Model yaşam döngüsünde şeffaflık, sürdürülebilirlik ve kayıtlı modeller.")

with kpi4:
    with st.container(border=True):
        st.markdown("### 🎯 Karar Destek")
        st.caption("Teknik ve idari birimlerin ortak çalışabildiği yapay zekâ çözümleri.")

st.divider()

# Kurumsal Tanım & Vizyon Bölümü (Executive Summary)
st.subheader("📌 Platform Vizyonu ve Amacı")

with st.container(border=True):
    st.markdown("""
    **AKSIS Framework**, kurum bünyesinde yürütülen makine öğrenmesi çalışmalarının **standartlaştırılması**, **merkezi olarak yönetilebilir hâle getirilmesi** ve kurumsal politika ile karar süreçlerinde daha etkin kullanılabilmesi amacıyla geliştirilen **modüler bir MLOps altyapısıdır**.

    ---
    
    #### 🎯 Temel Kazanımlar:
    * **⚡ Süreç Otomasyonu:** Veri analizi, veri sızıntısız (leakage-safe) ön işleme, model geliştirme, hiperparametre optimizasyonu ve değerlendirme süreçlerinin tek çatı altında otomatikleştirilmesi.
    * **🔄 Tekrar Üretilebilirlik & Sürdürülebilirlik:** Model yaşam döngüsünde uçtan uca izlenebilirlik sağlanarak yapay zekâ çözümlerinin kurumsal ölçekte yeniden kullanılabilir kılınması.
    * **🤝 Ortak Çalışma Kültürü:** Kullanıcı dostu arayüz ile teknik ve teknik olmayan birimlerin aynı altyapı üzerinden yapay zekâ tabanlı karar destek süreçlerine katılımının yaygınlaştırılması.
    """)

st.divider()

# Kurumsal Mimari & Pipeline Akışı
st.subheader("📐 Kurumsal Mimari & Pipeline Akışı")

tab_mimari, tab_akis, tab_guvenlik = st.tabs(["📊 Uçtan Uca Mimari", "🚀 4 Adımlı İş Akışı", "🛡️ Güvenlik & Standartlar"])

with tab_mimari:
    import os
    
    # Görsel dosya kontrolü
    assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
    possible_images = ["mimari.png", "mimari.jpg", "architecture.png", "architecture.jpg", "pipeline.png"]
    found_image = None
    
    for img_name in possible_images:
        img_path = os.path.join(assets_dir, img_name)
        if os.path.exists(img_path):
            found_image = img_path
            break
            
    if found_image:
        st.image(found_image, caption="AKSIS Kurumsal Platform Mimarisi & Pipeline Akışı", use_container_width=True)
    else:
        st.markdown("""
        ```mermaid
        flowchart LR
            subgraph DataLayer [📁 Veri & Katalog]
                D1[(Kurumsal Veri)] --> D2[DataSpec Doğrulama]
            end
            
            subgraph PipelineLayer [⚙️ AKSIS Çekirdek Pipeline]
                D2 --> P1[Önişleme & Sızıntı Önleme]
                P1 --> P2[Model & Algoritma Eğitimi]
                P2 --> P3[Optuna Hiperparametre Ayarı]
            end
            
            subgraph EvalLayer [📈 Değerlendirme & Dağıtım]
                P3 --> E1[Metrikler & Kafa Karışıklığı]
                E1 --> E2[Kayıtlı Model / Artifact]
                E2 --> E3[Toplu Çıkarım / Inference]
            end
            
            style DataLayer fill:#0F172A,stroke:#38BDF8,stroke-width:1px,color:#FFFFFF
            style PipelineLayer fill:#1E293B,stroke:#60A5FA,stroke-width:1px,color:#FFFFFF
            style EvalLayer fill:#0F172A,stroke:#34D399,stroke-width:1px,color:#FFFFFF
        ```
        """)
        st.info("💡 **Görsel Ekleme:** Kendi mimari şemanızı göstermek için görselinizi `frontend/assets/mimari.png` yoluna bırakmanız yeterlidir; sistem otomatik olarak burada görüntüleyecektir.")

with tab_akis:
    col_a, col_b = st.columns(2)
    with col_a:
        with st.container(border=True):
            st.markdown("#### 1. 📁 Veri Setlerini İnceleyin")
            st.write("Veri setlerinin satır, sütun ve eksik değer analizlerini inceleyin; veri setinin hangi görevlerle (Sınıflandırma, Regresyon, Anomali) uyumlu olduğunu belirleyin.")
            
        with st.container(border=True):
            st.markdown("#### 2. ⚙️ Deneyi Yapılandırın")
            st.write("Öğrenme türü, algoritma (XGBoost, CatBoost, Isolation Forest vb.), önişleme ve çapraz doğrulama stratejilerini seçerek konfigürasyonu tamamlayın.")

    with col_b:
        with st.container(border=True):
            st.markdown("#### 3. 📈 Eğitimi Başlatın & İzleyin")
            st.write("Model eğitimini asenkron olarak çalıştırın; canlı durum takibi ile Accuracy, F1-Score, RMSE veya Anomali Skor dağılımlarını anında gözlemleyin.")
            
        with st.container(border=True):
            st.markdown("#### 4. 🚀 Toplu Tahmin (Inference) Üretin")
            st.write("Eğitilen model çıktılarını kullanarak yeni kurumsal veri setleri üzerinde toplu çıkarımlar gerçekleştirin ve sonuçları raporlayın.")

with tab_guvenlik:
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("""
        ##### 🔒 Veri Güvenliği ve Gizliliği
        * Kurumsal veriler tamamen yerel/kurum sunucularında işlenir.
        * Dışarıya hiçbir veri veya model ağırlığı aktarılmaz.
        * Bağımsız REST API mimarisi ile kullanıcı arayüzü ve ML motoru izole çalışır.
        """)
    with col_g2:
        st.markdown("""
        ##### 📐 Metodolojik Doğruluk
        * **Leakage-Safe Preprocessing:** Eğitim ve test kümeleri arasında veri sızıntısını engelleyen pipeline mimarisi.
        * **Ground-Truth Denetimi:** Etiketsiz anomali analizlerinde yanıltıcı metrik üretimi engellenir.
        * **Tekrar Üretilebilirlik:** Her deney parametresi ve konfigürasyonu kayıt altına alınır.
        """)
