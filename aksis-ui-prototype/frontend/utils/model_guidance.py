"""
AKSIS Platformu - Model Rehberi ve Karar Destek Bilgi Tabanı
Algoritmaların güçlü, zayıf yönleri, display aliasları ve kullanım senaryoları.
"""

# Frontend Görüntüleme Eşleştirmeleri (Display Aliases)
ALGORITHM_DISPLAY_NAMES = {
    # Classification
    "logreg": "Lojistik Regresyon (logreg)",
    "random_forest_c": "Random Forest Sınıflandırıcı (random_forest_c)",
    "hgb_c": "HistGradientBoosting Sınıflandırıcı (hgb_c)",
    "svc": "Destek Vektör Sınıflandırıcı (svc)",
    "knn": "K-En Yakın Komşu (knn)",
    "catboost": "CatBoost (catboost)",
    "xgb_c": "XGBoost Sınıflandırıcı (xgb_c)",
    
    # Regression
    "ridge": "Ridge Regresyon (ridge)",
    "svr": "Destek Vektör Regresyon (svr)",
    "random_forest_r": "Random Forest Regresyon (random_forest_r)",
    "xgb": "XGBoost Regresyon (xgb)",
    "hgb_r": "HistGradientBoosting Regresyon (hgb_r)",
    
    # Anomaly Detection
    "isolation_forest": "Isolation Forest (isolation_forest)",
    "lof": "Local Outlier Factor (lof)",
    "one_class_svm": "One-Class SVM (one_class_svm)",
    "elliptic_envelope": "Elliptic Envelope (elliptic_envelope)",
    
    # Legacy / Friendly Name Compatibility
    "Logistic Regression": "Lojistik Regresyon (logreg)",
    "Random Forest": "Random Forest",
    "HistGradientBoosting": "HistGradientBoosting",
    "SVC": "SVC",
    "KNN": "KNN",
    "CatBoost": "CatBoost",
    "XGBoost": "XGBoost",
    "Ridge": "Ridge",
    "SVR": "SVR",
    "Isolation Forest": "Isolation Forest",
    "Local Outlier Factor": "Local Outlier Factor",
    "One-Class SVM": "One-Class SVM",
    "Elliptic Envelope": "Elliptic Envelope"
}

MODEL_GUIDANCE = {
    # --- Ağaç Tabanlı ve Boosting Modelleri ---
    "xgb_c": {
        "pros": "Tablosal verilerde endüstri standardı performans, gelişmiş regülarizasyon (L1/L2), yüksek hız ve esneklik.",
        "cons": "Hiperparametre ayarlarına duyarlıdır; kategorik veriler için ön kodlama gerektirir.",
        "best_for": "Yüksek doğruluk hedeflenen yarışma ve üretim seviyesi sınıflandırma modelleri."
    },
    "xgb": {
        "pros": "Regresyon problemlerinde yüksek kestirim gücü, gradyan artırma hızı, aykırı değer toleransı.",
        "cons": "Doğru parametre ayarı (learning rate, depth) yapılmadığında overfit riski.",
        "best_for": "Karmaşık ilişkiler içeren sayısal ve tablosal regresyon analizleri."
    },
    "catboost": {
        "pros": "Kategorik değişkenlerde üstün başarı, varsayılan parametrelerle yüksek doğruluk, aşırı öğrenmeye karşı yüksek direnç.",
        "cons": "Yüksek boyutlu seyrek (sparse) verilerde eğitimi görece daha uzun sürebilir.",
        "best_for": "Tablosal ve zengin kategorik sütunlar içeren kurumsal veri setleri."
    },
    "random_forest_c": {
        "pros": "Aykırı değerlere ve gürültüye karşı çok dayanıklı, aşırı öğrenme riski düşük, nitelik önem düzeylerini şeffaf sunar.",
        "cons": "Çok derin ağaçlarda bellek kullanımı artabilir; eğitim aralığının dışındaki değerlere genelleme yapamaz.",
        "best_for": "Dengeli, kararlı ve yorumlanabilir genel amaçlı sınıflandırma görevleri."
    },
    "random_forest_r": {
        "pros": "Doğrusal olmayan karmaşık ilişkileri başarıyla yakalar, aşırı uyum riski düşüktür.",
        "cons": "Eğitim kümesindeki minimum/maksimum hedef değerlerin ötesine ekstrapolasyon yapamaz.",
        "best_for": "Güvenilir ve kararlı tahminler gerektiren genel regresyon problemleri."
    },
    "hgb_c": {
        "pros": "Büyük veri setlerinde (10.000+ satır) olağanüstü hızlı eğitim, eksik değerleri (NaN) doğrudan işleyebilme.",
        "cons": "Çok küçük veri setlerinde aşırı uyum (overfit) riski taşıyabilir.",
        "best_for": "Büyük ölçekli veri setlerinde hızlı prototipleme ve iterasyon."
    },
    "hgb_r": {
        "pros": "Büyük regresyon verilerinde çok hızlı histogram tabanlı eğitim ve yerleşik eksik değer yönetimi.",
        "cons": "Düşük örneklemli veri setlerinde basit modellere göre dezavantajlı olabilir.",
        "best_for": "Geniş hacimli veri setlerinde hızlı regresyon modelleme."
    },
    
    # --- Doğrusal ve İstatistiksel Modeller ---
    "logreg": {
        "pros": "Çok hızlı eğitim ve tahmin, düşük işlem maliyeti, katsayıları sayesinde %100 şeffaf ve yorumlanabilir.",
        "cons": "Nitelikler arasındaki doğrusal olmayan karmaşık ilişkileri ve etkileşimleri tek başına yakalayamaz.",
        "best_for": "Açıklanabilirliğin ve hızın kritik olduğu yasal/regülatif karar destek süreçleri."
    },
    "ridge": {
        "pros": "L2 regülarizasyonu sayesinde çoklu doğrusal bağlantı (multicollinearity) sorununu çözer, kararlı katsayılar üretir.",
        "cons": "Yalnızca doğrusal ilişkileri modeller, değişken seçimi yapmaz.",
        "best_for": "Çok sayıda birbiriyle ilişkili sayısal sütun içeren regresyon analizleri."
    },
    
    # --- Kernel ve Komşuluk Tabanlı Modeller ---
    "svc": {
        "pros": "Yüksek boyutlu karmaşık uzaylarda etkilidir; net karar sınırları (decision boundary) çizer.",
        "cons": "Büyük veri setlerinde (N > 20.000) eğitim süresi ve bellek tüketimi çok artar; ölçeklendirmeye aşırı duyarlıdır.",
        "best_for": "Küçük/orta boyutlu ve karmaşık sınıflandırma problemleri."
    },
    "svr": {
        "pros": "Doğrusal olmayan regresyon ilişkilerini marjinal hata toleransı ile başarıyla öğrenir.",
        "cons": "Büyük veri setlerinde ölçeklenmesi zordur; hiperparametre ayarı hassasiyet gerektirir.",
        "best_for": "Küçük boyutlu ama karmaşık dalgalanmalar gösteren regresyon verileri."
    },
    "knn": {
        "pros": "Ön eğitim gerektirmez (lazy learning); karmaşık ve yerel karar sınırlarını sezgisel olarak öğrenir.",
        "cons": "Tahmin anında tüm veri setini taradığı için yavaştır; aykırı değerlere ve değişken ölçeklerine çok hassastır.",
        "best_for": "Benzerlik tabanlı karar verme ve düşük boyutlu kompakt veri kümeleri."
    },
    
    # --- Anomali Tespiti Modelleri ---
    "isolation_forest": {
        "pros": "Doğrusal zaman karmaşıklığı O(n) ile çok hızlıdır; veri dağılımı varsayımı yapmaz; yüksek boyutta etkilidir.",
        "cons": "Farklı yoğunluklardaki yerel kümelenme anomalilerini kaçırabilir.",
        "best_for": "Büyük ölçekli kurumsal log, işlem ve sahtekarlık (fraud) anomali taramaları."
    },
    "lof": {
        "pros": "Yerel veri yoğunluğunu temel alır; farklı yoğunluktaki kümelere ait yerel aykırı değerleri mükemmel yakalar.",
        "cons": "Büyük verilerde komşuluk hesabı maliyetlidir; yeni gelen veriler için anlık çıkarım zordur.",
        "best_for": "Kümelenmiş ve homojen olmayan veri dağılımlarında yerel anomali tespiti."
    },
    "one_class_svm": {
        "pros": "Normal verinin sınırlarını doğrusal olmayan kernel desteğiyle çok sıkı ve hassas bir şekilde çizer.",
        "cons": "Aykırı değer oranı parametresine çok duyarlıdır; büyük verilerde yavaşlar.",
        "best_for": "Temiz ve tek tip normal verinin bulunduğu nadir olay tespiti senaryoları."
    },
    "elliptic_envelope": {
        "pros": "Normal (Gaussian) dağılıma sahip verilerde teorik olarak en tutarlı ve hızlı anomali tespit yöntemidir.",
        "cons": "Verinin çok değişkenli normal dağıldığını varsayar; çarpık veya çok modlu verilerde yanıltıcı olabilir.",
        "best_for": "İyi temizlenmiş ve standart dağılıma uyan finansal/istatistiksel göstergeler."
    }
}

# Alias fallbacks for human readable keys
for k, v in list(MODEL_GUIDANCE.items()):
    clean_k = k.replace("_c", "").replace("_r", "").replace("_", " ").title()
    if clean_k not in MODEL_GUIDANCE:
        MODEL_GUIDANCE[clean_k] = v

def get_display_name(algorithm_id: str) -> str:
    """Algoritma kod adını kullanıcı dostu sunum etiketine dönüştürür."""
    if not algorithm_id:
        return ""
    return ALGORITHM_DISPLAY_NAMES.get(algorithm_id, str(algorithm_id))

def get_model_guidance(algorithm_name: str) -> dict:
    """Verilen algoritma için güçlü, zayıf yönleri ve en uygun kullanım senaryosunu döner."""
    if not algorithm_name:
        return {
            "pros": "Kurumsal makine öğrenmesi standartlarına uygun algoritma.",
            "cons": "Veri dağılımına ve hiperparametre seçimine göre performansı değişebilir.",
            "best_for": "Genel makine öğrenmesi görevleri."
        }
    
    # Try direct key or normalized key
    if algorithm_name in MODEL_GUIDANCE:
        return MODEL_GUIDANCE[algorithm_name]
    
    normalized = algorithm_name.lower().replace(" ", "_")
    if normalized in MODEL_GUIDANCE:
        return MODEL_GUIDANCE[normalized]
        
    return {
        "pros": "Kurumsal makine öğrenmesi standartlarına uygun genel amaçlı algoritma.",
        "cons": "Veri dağılımına ve hiperparametre seçimine göre performansı değişkenlik gösterebilir.",
        "best_for": "Genel kurumsal veri analitiği görevleri."
    }
