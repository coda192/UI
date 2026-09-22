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

            # 4.3. SAYISAL İSTATİSTİKLER GENEL BAKIŞ (Numeric Statistics Overview)
            st.markdown("#### 🔢 Sayısal İstatistikler Genel Bakış (Numeric Statistics Overview)")
            st.caption("Tüm sayısal sütunların karşılaştırmalı betimsel istatistikleri, çeyrek sınırları ve çarpıklık değerleri.")

            numeric_cols = [
                c for c in cols_list
                if (c.get("primitive_type") or "").lower() in ["numeric", "numerical"]
            ]

            if numeric_cols:
                numeric_table_rows = []
                for c in numeric_cols:
                    stats = c.get("statistics") or {}
                    cnt = stats.get("count")
                    mean_v = stats.get("mean")
                    std_v = stats.get("std")
                    min_v = stats.get("min")
                    q1_v = stats.get("25%")
                    med_v = stats.get("50%")
                    q3_v = stats.get("75%")
                    max_v = stats.get("max")
                    iqr_v = (q3_v - q1_v) if (q1_v is not None and q3_v is not None) else None
                    skew_v = stats.get("skewness")

                    numeric_table_rows.append({
                        "Sütun Adı": c.get("name"),
                        "Gözlem (Count)": f"{int(cnt):,}" if cnt is not None else "-",
                        "Ortalama (Mean)": f"{mean_v:.2f}" if mean_v is not None else "-",
                        "Std Sapma (Std)": f"{std_v:.2f}" if std_v is not None else "-",
                        "Min": f"{min_v:.2f}" if min_v is not None else "-",
                        "Q1 (%25)": f"{q1_v:.2f}" if q1_v is not None else "-",
                        "Medyan (%50)": f"{med_v:.2f}" if med_v is not None else "-",
                        "Q3 (%75)": f"{q3_v:.2f}" if q3_v is not None else "-",
                        "Maks": f"{max_v:.2f}" if max_v is not None else "-",
                        "IQR": f"{iqr_v:.2f}" if iqr_v is not None else "-",
                        "Çarpıklık (Skewness)": f"{skew_v:.3f}" if skew_v is not None else "Hesaplanamadı",
                    })

                df_numeric_stats = pd.DataFrame(numeric_table_rows)
                st.dataframe(df_numeric_stats, use_container_width=True)
            else:
                st.info("Veri setinde sayısal sütun bulunmamaktadır.")

            st.divider()

            # 4.4. SÜTUN GEZGİNİ (Column Explorer)
            st.markdown("#### 🔍 Sütun Gezgini (Column Explorer)")
            st.caption("Herhangi bir sütunu seçerek detaylı profilini, çeyrek aralıklarını ve çarpıklık yorumunu inceleyin.")

            if cols_list:
                col_names = [c.get("name") for c in cols_list]
                selected_col_name = st.selectbox(
                    "İncelenecek Sütunu Seçin",
                    options=col_names,
                    key=f"col_explorer_sel_{selected_id}"
                )

                selected_col = next((c for c in cols_list if c.get("name") == selected_col_name), None)

                if selected_col:
                    raw_prim = selected_col.get("primitive_type") or ""
                    prim_type = raw_prim.lower()

                    # ==========================================
                    # DURUM 1: SAYISAL KOLONLAR (Numeric)
                    # ==========================================
                    if prim_type in ["numeric", "numerical"]:
                        st.markdown(f"##### 🔢 Sayısal Sütun Analizi: `{selected_col_name}`")
                        stats = selected_col.get("statistics")
                        if stats:
                            # Metrik Kartları
                            sm1, sm2, sm3, sm4, sm5 = st.columns(5)
                            cnt = stats.get("count")
                            mean_v = stats.get("mean")
                            std_v = stats.get("std")
                            med_v = stats.get("50%")
                            q1_v = stats.get("25%")
                            q3_v = stats.get("75%")
                            iqr_v = (q3_v - q1_v) if (q1_v is not None and q3_v is not None) else None

                            with sm1:
                                st.metric("Gözlem Sayısı", f"{int(cnt):,}" if cnt is not None else "-")
                            with sm2:
                                st.metric("Ortalama", f"{mean_v:.2f}" if mean_v is not None else "-")
                            with sm3:
                                st.metric("Std Sapma", f"{std_v:.2f}" if std_v is not None else "-")
                            with sm4:
                                st.metric("Medyan (Q2)", f"{med_v:.2f}" if med_v is not None else "-")
                            with sm5:
                                st.metric("IQR (Q3 - Q1)", f"{iqr_v:.2f}" if iqr_v is not None else "-")

                            # Sayısal Analiz Sekmeleri
                            num_tab1, num_tab2, num_tab3 = st.tabs([
                                "📊 Çeyrek Dilimleri & IQR",
                                "📐 Çarpıklık (Skewness)",
                                "ℹ️ Dağılım & Histogram Bilgisi"
                            ])

                            with num_tab1:
                                # Çeyrek Görselleştirmesi (Quartile Visualization)
                                min_v = stats.get("min")
                                max_v = stats.get("max")

                                if all(v is not None for v in [min_v, q1_v, med_v, q3_v, max_v]):
                                    import plotly.graph_objects as go

                                    if min_v == max_v:
                                        st.info(f"ℹ️ **Sabit Değerli Sütun:** Tüm gözlemler ve çeyrek değerleri tek bir değere eşittir ({min_v:.2f}). Değişkenlik bulunmamaktadır.")
                                        fig_q = go.Figure()
                                        fig_q.add_vline(
                                            x=min_v,
                                            line_width=3,
                                            line_color="#4A90E2",
                                            annotation_text=f"Sabit Değer: {min_v:.2f}",
                                            annotation_position="top"
                                        )
                                        fig_q.update_layout(
                                            title=f"Sabit Değer Görselleştirmesi — {selected_col_name}",
                                            xaxis=dict(title=selected_col_name, range=[min_v - 1, min_v + 1]),
                                            yaxis=dict(showticklabels=False),
                                            height=200,
                                            margin=dict(t=40, b=20, l=10, r=10)
                                        )
                                        st.plotly_chart(fig_q, use_container_width=True)
                                    else:
                                        fig_q = go.Figure()

                                        # 4 Çeyrek Aralığı (Quartile Intervals)
                                        y_cat_q = "Çeyrek Dilimleri"
                                        y_cat_iqr = "IQR (Orta %50)"

                                        # Q0: Min -> Q1
                                        w_q0 = max(0.0, q1_v - min_v)
                                        hover_q0 = (
                                            f"<b>1. Çeyrek Dilimi (Q0: Min → Q1)</b><br>"
                                            f"Aralık: {min_v:.2f} — {q1_v:.2f}<br>"
                                            f"Aralık Genişliği: {w_q0:.2f}<br>"
                                            f"Pay: Gözlemlerin yaklaşık ilk %25'i<extra></extra>"
                                        )
                                        fig_q.add_trace(go.Bar(
                                            name="Q0: Min → Q1 (İlk %25)",
                                            y=[y_cat_q],
                                            x=[w_q0],
                                            base=[min_v],
                                            orientation='h',
                                            marker=dict(color="#4A90E2", line=dict(color="#1D60A5", width=1.5)),
                                            hovertemplate=hover_q0
                                        ))

                                        # Q1: Q1 -> Medyan
                                        w_q1 = max(0.0, med_v - q1_v)
                                        hover_q1 = (
                                            f"<b>2. Çeyrek Dilimi (Q1: Q1 → Medyan)</b><br>"
                                            f"Aralık: {q1_v:.2f} — {med_v:.2f}<br>"
                                            f"Aralık Genişliği: {w_q1:.2f}<br>"
                                            f"Pay: Gözlemlerin yaklaşık ikinci %25'i<extra></extra>"
                                        )
                                        fig_q.add_trace(go.Bar(
                                            name="Q1: Q1 → Medyan (İkinci %25)",
                                            y=[y_cat_q],
                                            x=[w_q1],
                                            base=[q1_v],
                                            orientation='h',
                                            marker=dict(color="#50E3C2", line=dict(color="#20A889", width=1.5)),
                                            hovertemplate=hover_q1
                                        ))

                                        # Q2: Medyan -> Q3
                                        w_q2 = max(0.0, q3_v - med_v)
                                        hover_q2 = (
                                            f"<b>3. Çeyrek Dilimi (Q2: Medyan → Q3)</b><br>"
                                            f"Aralık: {med_v:.2f} — {q3_v:.2f}<br>"
                                            f"Aralık Genişliği: {w_q2:.2f}<br>"
                                            f"Pay: Gözlemlerin yaklaşık üçüncü %25'i<extra></extra>"
                                        )
                                        fig_q.add_trace(go.Bar(
                                            name="Q2: Medyan → Q3 (Üçüncü %25)",
                                            y=[y_cat_q],
                                            x=[w_q2],
                                            base=[med_v],
                                            orientation='h',
                                            marker=dict(color="#F5A623", line=dict(color="#C47E0C", width=1.5)),
                                            hovertemplate=hover_q2
                                        ))

                                        # Q3: Q3 -> Maks
                                        w_q3 = max(0.0, max_v - q3_v)
                                        hover_q3 = (
                                            f"<b>4. Çeyrek Dilimi (Q3: Q3 → Maks)</b><br>"
                                            f"Aralık: {q3_v:.2f} — {max_v:.2f}<br>"
                                            f"Aralık Genişliği: {w_q3:.2f}<br>"
                                            f"Pay: Gözlemlerin yaklaşık son %25'i<extra></extra>"
                                        )
                                        fig_q.add_trace(go.Bar(
                                            name="Q3: Q3 → Maks (Dördüncü %25)",
                                            y=[y_cat_q],
                                            x=[w_q3],
                                            base=[q3_v],
                                            orientation='h',
                                            marker=dict(color="#E94E77", line=dict(color="#B32448", width=1.5)),
                                            hovertemplate=hover_q3
                                        ))

                                        # IQR Barı
                                        w_iqr = max(0.0, q3_v - q1_v)
                                        hover_iqr = (
                                            f"<b>IQR (Çeyrekler Açıklığı: Q1 → Q3)</b><br>"
                                            f"Aralık: {q1_v:.2f} — {q3_v:.2f}<br>"
                                            f"IQR Genişliği: {w_iqr:.2f}<br>"
                                            f"Pay: Gözlemlerin orta %50'si<extra></extra>"
                                        )
                                        fig_q.add_trace(go.Bar(
                                            name="IQR (Q1 → Q3)",
                                            y=[y_cat_iqr],
                                            x=[w_iqr],
                                            base=[q1_v],
                                            orientation='h',
                                            marker=dict(color="#9013FE", line=dict(color="#6005B5", width=1.5)),
                                            hovertemplate=hover_iqr
                                        ))

                                        # Sınır Çizgileri ve Etiketler
                                        bounds = [
                                            ("Min", min_v),
                                            ("Q1", q1_v),
                                            ("Medyan", med_v),
                                            ("Q3", q3_v),
                                            ("Maks", max_v)
                                        ]
                                        grouped_bounds = {}
                                        for b_name, b_val in bounds:
                                            grouped_bounds.setdefault(b_val, []).append(b_name)

                                        for b_val, b_names in grouped_bounds.items():
                                            lbl = f"{' & '.join(b_names)}: {b_val:.2f}"
                                            fig_q.add_vline(
                                                x=b_val,
                                                line_width=1.5,
                                                line_dash="dash",
                                                line_color="rgba(128, 128, 128, 0.6)",
                                                annotation_text=lbl,
                                                annotation_position="top",
                                                annotation_font_size=10
                                            )

                                        span = max_v - min_v
                                        pad = span * 0.08 if span > 0 else 1.0
                                        fig_q.update_layout(
                                            barmode='overlay',
                                            title=f"Çeyrek Dilimleri ve IQR Görselleştirmesi — {selected_col_name}",
                                            xaxis=dict(
                                                title=f"{selected_col_name} Değer Ekseni",
                                                range=[min_v - pad, max_v + pad]
                                            ),
                                            yaxis=dict(title=""),
                                            height=290,
                                            margin=dict(t=50, b=30, l=10, r=10),
                                            legend=dict(orientation="h", yanchor="bottom", y=-0.45, xanchor="center", x=0.5)
                                        )
                                        st.plotly_chart(fig_q, use_container_width=True)
                                else:
                                    st.info("Bu sütun için beşli özet (Min, Q1, Medyan, Q3, Maks) değerlerinin tümü mevcut değildir.")

                            with num_tab2:
                                # Çarpıklık (Skewness) Yorumu
                                skew_v = stats.get("skewness")

                                if skew_v is not None:
                                    disp_skew = f"{skew_v:.3f}"
                                    if skew_v > 0.05:
                                        interp_msg = "Pozitif çarpıklık: Dağılımın sağ kuyruğu daha uzun veya daha ağırdır (sağa çarpık)."
                                    elif skew_v < -0.05:
                                        interp_msg = "Negatif çarpıklık: Dağılımın sol kuyruğu daha uzun veya daha ağırdır (sola çarpık)."
                                    else:
                                        interp_msg = "Sıfıra yakın çarpıklık: Bu ölçüme göre dağılımda görece sınırlı bir asimetri bulunmaktadır."
                                else:
                                    disp_skew = "Hesaplanamadı"
                                    interp_msg = "Çarpıklık değeri bu sütun için hesaplanamadı veya veri setinde mevcut değildir."

                                sk1, sk2 = st.columns([3, 7])
                                with sk1:
                                    st.metric("Çarpıklık Değeri", disp_skew)
                                with sk2:
                                    st.info(f"**Yorum:** {interp_msg}")

                                st.caption(
                                    "ℹ️ *Çarpıklık (skewness), bir dağılımın asimetrisini tanımlarken; "
                                    "çeyrekler (quartiles), değerlerin dağılım içindeki konumlarını ve yayılımını gösterir. "
                                    "Sıfıra yakın çarpıklık değeri normal dağılımın kesin bir kanıtı değildir.*"
                                )

                            with num_tab3:
                                st.info(
                                    "ℹ️ **Histogram & Kutu Grafiği Durumu:**\n\n"
                                    "- **Çeyrek & IQR Analizi:** Aktif (API `statistics` beşli özeti üzerinden sunulmaktadır).\n"
                                    "- **Kutu Grafiği (Box Plot):** API ham gözlem verisi ve aykırı değer noktalarını taşımadığından, dağılım yayılımı Çeyrek Dilimleri & IQR ekseni ile temsil edilmektedir.\n"
                                    "- **Histogram:** Mevcut `GET /api/v1/datasets/{dataset_id}/info` endpoint'i özet istatistikleri sunmakta olup, ham veri dağılımını veya histogram kutularını (bins) sağlayan ayrı bir histogram endpoint'i mevcut backend sözleşmesinde tanımlı değildir."
                                )
                        else:
                            st.warning("Bu sayısal sütun için detaylı betimsel istatistikler (statistics) bulunmamaktadır.")

                    # ==========================================
                    # DURUM 2: KATEGORİK KOLONLAR (Categorical)
                    # ==========================================
                    elif prim_type == "categorical":
                        st.markdown(f"##### 🏷️ Kategorik Sütun Profili & Dağılımı: `{selected_col_name}`")
                        np1, np2, np3, np4 = st.columns(4)
                        with np1:
                            st.metric("Veri Tipi (dtype)", selected_col.get("dtype") or "-")
                        with np2:
                            st.metric("Alt Tip", selected_col.get("subtype") or "-")
                        with np3:
                            st.metric("Benzersiz Değer", f"{selected_col.get('unique_count', 0):,}")
                        with np4:
                            st.metric("Eksik Değer Oranı", f"%{(selected_col.get('missing_rate', 0.0) * 100):.2f}")

                        if selected_col.get("flags"):
                            st.write(f"**Etiketler (Flags):** {', '.join(selected_col.get('flags'))}")

                        # Top-K Kategori Dağılımı Görselleştirmesi
                        top_cats = selected_col.get("top_categories") or []
                        if top_cats:
                            st.markdown("###### 📊 Kategori Frekans Dağılımı (Top-K Distribution)")

                            # Top-5 / Top-10 Seçici
                            k_choice = st.radio(
                                "Gösterilecek Kategori Sayısı:",
                                options=["Top-5", "Top-10"],
                                horizontal=True,
                                key=f"topk_sel_{selected_id}_{selected_col_name}"
                            )
                            k_limit = 5 if k_choice == "Top-5" else 10
                            displayed_cats = top_cats[:k_limit]

                            # Geçerli gözlem sayısı (valid_count) kontrolü
                            valid_cnt = selected_col.get("valid_count")
                            if valid_cnt is None:
                                row_cnt = info.get("row_count", 0)
                                missing_cnt = selected_col.get("missing_count", 0)
                                valid_cnt = max(0, row_cnt - missing_cnt)

                            if valid_cnt == 0:
                                st.info("Bu sütunda geçerli (eksik olmayan) gözlem bulunmamaktadır.")
                            else:
                                displayed_sum = 0
                                cat_records = []

                                for tc in displayed_cats:
                                    if hasattr(tc, "value"):
                                        c_val = tc.value
                                        c_count = tc.count
                                    elif isinstance(tc, dict):
                                        c_val = tc.get("value", "")
                                        c_count = tc.get("count", 0)
                                    else:
                                        continue

                                    displayed_sum += c_count
                                    pct = (c_count / valid_cnt * 100) if valid_cnt > 0 else 0.0
                                    cat_records.append({
                                        "Kategori": str(c_val),
                                        "Frekans": c_count,
                                        "Yüzde": f"%{pct:.1f}",
                                        "Oran": pct,
                                        "Grup": "Kategori"
                                    })

                                # Dinamik Kalan (Remainder) Hesaplama:
                                # other_count_for_selection = valid_count - sum(displayed category counts)
                                other_count_for_selection = max(0, valid_cnt - displayed_sum)

                                if other_count_for_selection > 0:
                                    other_pct = (other_count_for_selection / valid_cnt * 100) if valid_cnt > 0 else 0.0
                                    cat_records.append({
                                        "Kategori": "Diğer",
                                        "Frekans": other_count_for_selection,
                                        "Yüzde": f"%{other_pct:.1f}",
                                        "Oran": other_pct,
                                        "Grup": "Kalan"
                                    })

                                # Kapsama Oranı (Coverage Ratio) ve Temsil İlerleme Çubuğu
                                displayed_cats_sum = displayed_sum
                                coverage_pct = min(100.0, (displayed_cats_sum / valid_cnt * 100)) if valid_cnt > 0 else 0.0

                                st.progress(min(1.0, coverage_pct / 100.0))
                                cov_col1, cov_col2 = st.columns([7, 3])
                                with cov_col1:
                                    st.caption(f"🎯 **Temsil Gücü:** Seçilen {k_choice} kategori, toplam geçerli gözlemlerin **%{coverage_pct:.1f}**'ini kapsamaktadır.")
                                with cov_col2:
                                    first_pct = (displayed_cats[0].count if hasattr(displayed_cats[0], 'count') else displayed_cats[0].get('count', 0)) / valid_cnt if valid_cnt > 0 else 0.0
                                    if first_pct > 0.5:
                                        st.caption("⚠️ **Yüksek Sınıf Konsantrasyonu** (Baskın Sınıf)")
                                    else:
                                        st.caption("✅ **Dengeli Sınıf Dağılımı**")

                                # Özet Metrikler
                                tc_m1, tc_m2, tc_m3 = st.columns(3)
                                with tc_m1:
                                    st.metric("Geçerli Gözlem (Valid)", f"{valid_cnt:,}")
                                with tc_m2:
                                    st.metric(f"Listelenen ({k_choice})", f"{len(displayed_cats)} kategori")
                                with tc_m3:
                                    if other_count_for_selection > 0:
                                        st.metric("Diğer (Kalan Pay)", f"{other_count_for_selection:,} (%{other_pct:.1f})")
                                    else:
                                        st.metric("Diğer (Kalan Pay)", "0 (%0.0)")

                                # İkili Görselleştirme: Sol Yatay Bar + Sağ Donut
                                if cat_records:
                                    import plotly.graph_objects as go

                                    # Modern Gradient Palet
                                    palette = [
                                        "#1565C0", "#1976D2", "#1E88E5", "#2196F3", "#42A5F5",
                                        "#26A69A", "#00897B", "#00796B", "#FFB300", "#FB8C00"
                                    ]

                                    vis_col1, vis_col2 = st.columns([6, 4])

                                    # 1. SOL: Degrade Yatay Bar Grafiği
                                    with vis_col1:
                                        chart_records = cat_records[::-1]
                                        y_vals = [r["Kategori"] for r in chart_records]
                                        x_vals = [r["Frekans"] for r in chart_records]
                                        hover_texts = [
                                            f"<b>{r['Kategori']}</b><br>Frekans: {r['Frekans']:,}<br>Pay: {r['Yüzde']}<extra></extra>"
                                            for r in chart_records
                                        ]
                                        bar_colors = [
                                            "#90A4AE" if r["Grup"] == "Kalan" else palette[i % len(palette)]
                                            for i, r in enumerate(chart_records)
                                        ]

                                        fig_bar = go.Figure(go.Bar(
                                            y=y_vals,
                                            x=x_vals,
                                            orientation='h',
                                            text=[f"{v:,} ({r['Yüzde']})" for v, r in zip(x_vals, chart_records)],
                                            textposition='outside',
                                            hovertemplate=hover_texts,
                                            marker=dict(
                                                color=bar_colors,
                                                line=dict(color='rgba(255, 255, 255, 0.6)', width=1)
                                            )
                                        ))
                                        fig_bar.update_layout(
                                            title=f"Frekans Çubuk Grafiği — {selected_col_name}",
                                            xaxis=dict(title="Gözlem Adedi", showgrid=True, gridcolor='rgba(200,200,200,0.2)'),
                                            yaxis=dict(title=""),
                                            height=max(240, len(chart_records) * 38),
                                            margin=dict(t=40, b=20, l=10, r=80)
                                        )
                                        st.plotly_chart(fig_bar, use_container_width=True)

                                    # 2. SAĞ: Merkezi Metrikli Donut (Halka) Grafiği
                                    with vis_col2:
                                        donut_labels = [r["Kategori"] for r in cat_records]
                                        donut_vals = [r["Frekans"] for r in cat_records]
                                        donut_colors = [
                                            "#90A4AE" if r["Grup"] == "Kalan" else palette[i % len(palette)]
                                            for i, r in enumerate(cat_records)
                                        ]

                                        fig_donut = go.Figure(data=[go.Pie(
                                            labels=donut_labels,
                                            values=donut_vals,
                                            hole=0.55,
                                            marker=dict(colors=donut_colors, line=dict(color='#FFFFFF', width=2)),
                                            textinfo="percent",
                                            hoverinfo="label+value+percent",
                                            hovertemplate="<b>%{label}</b><br>Adet: %{value:,}<br>Oran: %{percent}<extra></extra>"
                                        )])
                                        fig_donut.update_layout(
                                            title="Yüzdesel Dağılım",
                                            annotations=[dict(
                                                text=f"<b>{valid_cnt:,}</b><br><span style='font-size:11px;color:#78909C'>Geçerli</span>",
                                                x=0.5, y=0.5, font_size=15, showarrow=False
                                            )],
                                            height=max(240, len(chart_records) * 38),
                                            margin=dict(t=40, b=10, l=10, r=10),
                                            legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5)
                                        )
                                        st.plotly_chart(fig_donut, use_container_width=True)

                                    # 3. Kümülatif Dağılım & Pareto Tablosu (Expander)
                                    with st.expander("📋 Detaylı Kategori ve Kümülatif Pay Tablosu"):
                                        cum_sum = 0
                                        pareto_rows = []
                                        for r in cat_records:
                                            cum_sum += r["Frekans"]
                                            cum_pct = (cum_sum / valid_cnt * 100) if valid_cnt > 0 else 0.0
                                            pareto_rows.append({
                                                "Kategori": r["Kategori"],
                                                "Tür": r["Grup"],
                                                "Frekans": f"{r['Frekans']:,}",
                                                "Tekil Pay (%)": r["Yüzde"],
                                                "Kümülatif Pay (%)": f"%{cum_pct:.1f}"
                                            })
                                        st.dataframe(pd.DataFrame(pareto_rows), use_container_width=True)
                        else:
                            st.info("Bu kategorik sütun için en sık gözlenen kategoriler (top_categories) bulunmamaktadır.")

                    # ==========================================
                    # DURUM 3: DİĞER TÜRLER (Boolean, Datetime, ID, Constant, Text, Unknown vb.)
                    # ==========================================
                    else:
                        type_display = raw_prim.capitalize() if raw_prim else "Bilinmeyen Tip"
                        st.markdown(f"##### 📋 {type_display} Sütun Profili: `{selected_col_name}`")
                        st.caption(f"Bu sütunun primitif tipi `{raw_prim or 'unknown'}` olarak tanımlanmıştır. Bu tip için ek sayısal veya kategorik analiz seçeneği bulunmamaktadır.")

                        np1, np2, np3, np4 = st.columns(4)
                        with np1:
                            st.metric("Veri Tipi (dtype)", selected_col.get("dtype") or "-")
                        with np2:
                            st.metric("Algılanan Tip / Alt Tip", f"{raw_prim or '-'} / {selected_col.get('subtype') or '-'}")
                        with np3:
                            st.metric("Benzersiz Değer", f"{selected_col.get('unique_count', 0):,}")
                        with np4:
                            st.metric("Eksik Değer Oranı", f"%{(selected_col.get('missing_rate', 0.0) * 100):.2f}")

                        if selected_col.get("flags"):
                            st.write(f"**Etiketler (Flags):** {', '.join(selected_col.get('flags'))}")

            st.divider()

            # 4.5. PEARSON KORELASYON ANALİZİ (Pearson Correlation Matrix)
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

