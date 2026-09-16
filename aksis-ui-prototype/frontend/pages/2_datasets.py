import streamlit as st
import pandas as pd
from api.client import AksisAPIError

st.title("📁 Veri Setleri")
st.caption("AKSIS Çerçevesinde kayıtlı veri setleri, şemalar ve teknik meta veriler.")

client = st.session_state.client

TASK_NAMES_TR = {
    "classification": "Sınıflandırma",
    "regression": "Regresyon",
    "anomaly_detection": "Anomali Tespiti"
}

try:
    datasets = client.get_datasets()
    
    if not datasets:
        st.info("Kullanılabilir veri seti bulunamadı.")
        st.stop()
        
    dataset_options = {
        d["id"]: f"{d.get('display_name')} ({d['id']})" if d.get("display_name") else d["id"]
        for d in datasets
    }
    
    # Oturum durumunda önceden seçili veri seti varsa onu kullan
    default_idx = 0
    if "selected_dataset_id" in st.session_state:
        ids = list(dataset_options.keys())
        if st.session_state.selected_dataset_id in ids:
            default_idx = ids.index(st.session_state.selected_dataset_id)
            
    selected_id = st.selectbox(
        "İncelenecek Veri Setini Seçin", 
        options=list(dataset_options.keys()), 
        format_func=lambda x: dataset_options[x],
        index=default_idx
    )
    
    if selected_id:
        st.session_state.selected_dataset_id = selected_id
        
        ds = client.get_dataset(selected_id)
        
        # 1. BAŞLIK VE HİYERARŞİ (DataSpec Display Name ve Raw ID)
        display_title = ds.get("display_name") or ds.get("id")
        raw_id = ds.get("id")
        
        st.markdown(f"## 📁 {display_title}")
        if ds.get("display_name") and ds.get("display_name") != raw_id:
            st.caption(f"`{raw_id}`")
            
        # 2. AÇIKLAMA BÖLÜMÜ (Doğrudan Başlığın Altında)
        desc = ds.get("description")
        if desc:
            st.write(desc)
        else:
            st.caption("Açıklama sağlanmamış.")
            
        st.divider()
        
        # 3. STATİK METAVERİ VE TEKNİK DETAYLAR (DataSpec Alanları)
        st.subheader("⚙️ Veri Seti Tanımı & Teknik Özellikler")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            target_val = ds.get("target")
            st.metric("Hedef Değişken (Target)", target_val if target_val else "Yok (Etiketsiz)")
        with col2:
            source_val = ds.get("source")
            st.metric("Veri Kaynağı (Source)", source_val if source_val else "Belirtilmemiş")
        with col3:
            local_val = ds.get("local_data")
            if local_val is True:
                local_str = "Evet (Yerel)"
            elif local_val is False:
                local_str = "Hayır (Uzak / Stream)"
            else:
                local_str = "Belirtilmemiş"
            st.metric("Yerel Veri (Local Data)", local_str)
            
        if ds.get("columns_to_use"):
            st.write(f"**Kullanılacak Sütunlar (Columns to Use):** {', '.join(ds.get('columns_to_use'))}")
            
        if ds.get("compatible_tasks"):
            comp_tasks = [TASK_NAMES_TR.get(t, t) for t in ds.get("compatible_tasks", [])]
            st.write(f"**Uyumlu Görevler:** {', '.join(comp_tasks)}")
            
        if ds.get("identifier_columns"):
            st.write(f"**Kimlik / ID Sütunları:** {', '.join(ds.get('identifier_columns'))}")
            
        st.divider()
        
        # 4. VERİ SETİ ANALİZİ (Dataset Analysis)
        st.subheader("🔬 Veri Seti Analizi (Dataset Analysis)")
        
        btn_col1, btn_col2 = st.columns([8, 2])
        with btn_col2:
            refresh_clicked = st.button("🔄 Analizi Yenile", key=f"btn_refresh_{selected_id}", use_container_width=True)
            
        with st.spinner("Veri seti analizi yükleniyor..."):
            try:
                info = client.get_dataset_info(selected_id, refresh=refresh_clicked)
            except AksisAPIError as e:
                st.error(f"Veri analizi alınamadı: {str(e)}")
                info = None

        if info:
            # 4.1. ÖZET METRİK KARTLARI (Summary Cards)
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Satır Sayısı (Rows)", f"{info.get('row_count', 0):,}")
            with m2:
                st.metric("Sütun Sayısı (Columns)", f"{info.get('column_count', 0):,}")
            with m3:
                st.metric("Toplam Eksik Değer", f"{info.get('missing_value_count', 0):,}")
            with m4:
                st.metric("Eksik Değerli Sütun", f"{info.get('missing_column_count', 0):,}")
            
            # Sütun Tipi Dağılımı (Type Distribution)
            type_counts = info.get("type_counts", {})
            if type_counts:
                import plotly.express as px
                fig_types = px.pie(
                    names=list(type_counts.keys()),
                    values=list(type_counts.values()),
                    hole=0.4,
                    title="Algılanan Sütun Tipi Dağılımı",
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
                fig_types.update_traces(textposition="inside", textinfo="percent+label")
                fig_types.update_layout(height=260, margin=dict(t=35, b=10, l=10, r=10))
                st.plotly_chart(fig_types, use_container_width=True)

            st.divider()

            # 4.2. SÜTUN PROFİLLERİ (Column Profiles)
            st.markdown("#### 📋 Sütun Profilleri (Column Profiles)")
            cols_list = info.get("columns", [])
            if cols_list:
                df_profile_cols = pd.DataFrame([
                    {
                        "Sütun Adı": c.get("name"),
                        "Algılanan Tip": c.get("primitive_type"),
                        "Alt Tip": c.get("subtype") or "-",
                        "Veri Tipi": c.get("dtype"),
                        "Benzersiz Değer": f"{c.get('unique_count', 0):,}",
                        "Eksik Değer": f"{c.get('missing_count', 0):,}",
                        "Eksik Oranı (%)": f"{(c.get('missing_rate', 0.0) * 100):.2f}%",
                        "Etiketler (Flags)": ", ".join(c.get("flags", [])) if c.get("flags") else "-"
                    }
                    for c in cols_list
                ])
                st.dataframe(df_profile_cols, use_container_width=True)
                
                # Eksik Değer Grafiği (Yalnızca eksik değer içeren sütunlar)
                missing_cols = [c for c in cols_list if c.get("missing_count", 0) > 0]
                if missing_cols:
                    import plotly.express as px
                    df_missing = pd.DataFrame([
                        {
                            "Sütun": c.get("name"),
                            "Eksik Sayısı": c.get("missing_count", 0),
                            "Eksik Oranı (%)": round(c.get("missing_rate", 0.0) * 100, 2)
                        }
                        for c in missing_cols
                    ]).sort_values(by="Eksik Sayısı", ascending=True)

                    fig_missing = px.bar(
                        df_missing,
                        x="Eksik Sayısı",
                        y="Sütun",
                        orientation="h",
                        text="Eksik Sayısı",
                        hover_data=["Eksik Oranı (%)"],
                        color="Eksik Oranı (%)",
                        color_continuous_scale="Reds",
                        title="Eksik Değer Dağılımı (Yalnızca Eksik Değer İçeren Sütunlar)"
                    )
                    fig_missing.update_layout(height=max(180, len(missing_cols) * 35), margin=dict(t=35, b=10, l=10, r=10))
                    st.plotly_chart(fig_missing, use_container_width=True)
                else:
                    st.caption("✅ Veri setinde hiçbir sütunda eksik değer bulunmamaktadır.")
            else:
                st.caption("Sütun profil bilgisi bulunamadı.")

            st.divider()

            # 4.3. PEARSON KORELASYON ANALİZİ (Pearson Correlation Matrix)
            st.markdown("#### 📈 Pearson Korelasyon Matrisi (Pearson Correlation Matrix)")
            corr = info.get("correlation", {})
            corr_cols = corr.get("columns", [])
            corr_matrix = corr.get("matrix", [])

            if corr_cols and corr_matrix and len(corr_cols) > 1:
                import plotly.graph_objects as go
                text_matrix = [
                    [f"{val:.2f}" if val is not None else "-" for val in row]
                    for row in corr_matrix
                ]
                fig_corr = go.Figure(
                    data=go.Heatmap(
                        z=corr_matrix,
                        x=corr_cols,
                        y=corr_cols,
                        text=text_matrix,
                        texttemplate="%{text}",
                        textfont={"size": 10},
                        colorscale="RdBu_r",
                        zmin=-1.0,
                        zmax=1.0,
                        hoverongaps=False,
                        hovertemplate="<b>%{y}</b> ile <b>%{x}</b><br>Korelasyon: %{text}<extra></extra>"
                    )
                )
                fig_corr.update_layout(
                    title="Pearson Korelasyon Isı Haritası",
                    height=max(360, len(corr_cols) * 45),
                    margin=dict(t=40, b=10, l=10, r=10)
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("Korelasyon matrisi için yeterli sayıda sayısal sütun bulunmuyor.")

except AksisAPIError as e:
    st.error(f"Hata: {str(e)}")

