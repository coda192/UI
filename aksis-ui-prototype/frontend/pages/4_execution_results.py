import streamlit as st
import time
from api.client import AksisAPIError
import pandas as pd

st.title("📈 Çalıştırma & Sonuçlar")

client = st.session_state.client

STATUS_TR = {
    "configured": "⚙️ Yapılandırıldı (Eğitime Hazır)",
    "running": "⏳ Eğitiliyor...",
    "completed": "✅ Tamamlandı",
    "failed": "❌ Başarısız"
}

SUMMARY_KEYS_TR = {
    "total_samples": "Toplam Örnek Sayısı",
    "normal_samples": "Normal Örnek Sayısı",
    "detected_anomalies": "Tespit Edilen Anomali",
    "anomaly_rate": "Anomali Oranı"
}

TABLE_TITLES_TR = {
    "confusion_matrix": "Hata / Karışıklık Matrisi (Confusion Matrix)",
    "top_anomalies": "En Yüksek Skorlu Anomaliler",
    "anomalous_records": "Tespit Edilen Anomali Kayıtları"
}

try:
    experiments = client.list_experiments()
    if not experiments:
        st.info("Kayıtlı deney bulunamadı. Lütfen önce 'Deney Yapılandırma' sayfasından yeni bir deney oluşturun.")
        st.stop()
        
    exp_options = {e["id"]: f"{e['name']} - [{STATUS_TR.get(e['status'], e['status'])}]" for e in experiments}
    
    default_idx = 0
    if "last_experiment_id" in st.session_state:
        ids = list(exp_options.keys())
        if st.session_state.last_experiment_id in ids:
            default_idx = ids.index(st.session_state.last_experiment_id)
            
    selected_exp_id = st.selectbox(
        "İncelenecek veya Çalıştırılacak Deneyi Seçin", 
        options=list(exp_options.keys()), 
        format_func=lambda x: exp_options[x], 
        index=default_idx
    )
    st.session_state.last_experiment_id = selected_exp_id
    
    meta = client.get_experiment(selected_exp_id)
    status = meta.get("status")
    
    st.write(f"**Güncel Durum:** {STATUS_TR.get(status, status)}")
    
    if status == "configured":
        if st.button("🚀 Eğitimi Başlat (Run Experiment)"):
            client.run_experiment(selected_exp_id)
            st.success("Eğitim süreci başlatıldı...")
            st.rerun()
            
    elif status == "running":
        progress = st.progress(0)
        status_text = st.empty()
        
        # Durum sorgulama (Polling)
        for i in range(15):
            meta = client.get_experiment(selected_exp_id)
            if meta["status"] != "running":
                break
            status_text.text(f"Model eğitiliyor, lütfen bekleyin... ({i+1}s)")
            progress.progress(min((i+1)*10, 100))
            time.sleep(1)
            
        st.rerun()
        
    elif status == "completed":
        st.success("✅ Model eğitimi ve değerlendirme süreci başarıyla tamamlandı.")
        
        results = client.get_experiment_results(selected_exp_id)
        
        st.header("📊 Değerlendirme & Sonuç Özeti")
        summary = results.get("summary", {})
        if summary:
            cols = st.columns(len(summary))
            for col, (k, v) in zip(cols, summary.items()):
                label = SUMMARY_KEYS_TR.get(k, k.replace('_', ' ').title())
                col.metric(label, f"{v:,}" if isinstance(v, (int, float)) and v > 100 else v)
                
        metrics = results.get("metrics", {})
        
        # Göreve göre metrikleri listele
        task = results.get("task")
        has_gt = results.get("has_ground_truth", False)
        
        if task == "classification":
            st.subheader("Sınıflandırma Başarı Metrikleri")
            cls_metrics = metrics.get("classification_metrics", {})
            mcols = st.columns(4)
            for i, (k, v) in enumerate(cls_metrics.items()):
                mcols[i % 4].metric(k.upper(), f"{v:.4f}" if isinstance(v, float) else v)
                
        elif task == "regression":
            st.subheader("Regresyon Hata & Başarı Metrikleri")
            reg_metrics = metrics.get("regression_metrics", {})
            mcols = st.columns(3)
            for i, (k, v) in enumerate(reg_metrics.items()):
                mcols[i % 3].metric(k.upper(), f"{v:.4f}" if isinstance(v, float) else v)
                
        elif task == "anomaly_detection":
            if has_gt:
                st.subheader("Etiketli Anomali Metrikleri (Ground Truth)")
                anom_metrics = metrics.get("anomaly_metrics", {})
                mcols = st.columns(3)
                for i, (k, v) in enumerate(anom_metrics.items()):
                    mcols[i % 3].metric(k.upper(), f"{v:.4f}" if isinstance(v, float) else v)
            else:
                st.info("💡 **Etiketsiz Anomali Tespiti:** Veri setinde gerçek etiket (ground truth) bulunmadığı için denetimli doğruluk/F1 metrikleri hesaplanmamıştır.")
                
        st.header("📈 Görselleştirmeler ve Tablolar")
        
        tables = results.get("tables", {})
        for table_name, data in tables.items():
            label = TABLE_TITLES_TR.get(table_name, table_name.replace('_', ' ').title())
            st.write(f"**{label}**")
            st.dataframe(pd.DataFrame(data), use_container_width=True)
            
        visualizations = results.get("visualizations", [])
        for viz in visualizations:
            title = viz.get("title", viz.get("type"))
            with st.expander(f"📊 {title}", expanded=True):
                html_content = viz.get("html_content")
                if html_content:
                    # Plotly HTML içeriğini tam interaktif olarak ekrana göm
                    import streamlit.components.v1 as components
                    components.html(html_content, height=420, scrolling=False)
                elif viz.get("data"):
                    st.json(viz.get("data", {}))
                
except AksisAPIError as e:
    st.error(f"Hata: {str(e)}")
