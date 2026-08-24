"""
AKSIS Platformu - Model Rehberi ve Karar Destek Bilgi Tabanı
Algoritmaların güçlü, zayıf yönleri ve en uygun kullanım senaryoları.
"""

MODEL_GUIDANCE = {
    # --- Sınıflandırma ve Regresyon Ortak / Ağaç Tabanlı Modeller ---
    "CatBoost": {
        "pros": "Kategorik değişkenlerde üstün başarı, varsayılan parametrelerle yüksek doğruluk, aşırı öğrenmeye (overfitting) karşı yüksek direnç.",
        "cons": "Yüksek boyutlu seyrek (sparse) verilerde eğitimi görece daha uzun sürebilir.",
        "best_for": "Tablosal ve zengin kategorik sütunlar içeren kurumsal veri setleri."
    },
    "XGBoost": {
        "pros": "Tablosal verilerde endüstri standardı performans, gelişmiş regülarizasyon (L1/L2), yüksek hız ve esneklik.",
        "cons": "Hiperparametre ayarlarına duyarlıdır; kategorik veriler için ön kodlama gerektirir.",
        "best_for": "Yüksek doğruluk hedeflenen yarışma ve üretim seviyesi tahmin modelleri."
    },
    "Random Forest": {
        "pros": "Aykırı değerlere ve gürültüye karşı çok dayanıklı, aşırı öğrenme riski düşük, nitelik önem düzeylerini (Feature Importance) şeffaf sunar.",
        "cons": "Çok derin ağaçlarda bellek kullanımı artabilir; eğitim aralığının dışındaki değerlere genelleme yapamaz.",
        "best_for": "Dengeli, kararlı ve yorumlanabilir genel amaçlı makine öğrenmesi görevleri."
    },
    "HistGradientBoosting": {
        "pros": "Büyük veri setlerinde (10.000+ satır) olağanüstü hızlı eğitim, eksik değerleri (NaN) doğrudan işleyebilme.",
        "cons": "Çok küçük veri setlerinde aşırı uyum (overfit) riski taşıyabilir.",
        "best_for": "Büyük ölçekli veri setlerinde hızlı prototipleme ve iterasyon."
    },
    
    # --- Doğrusal ve İstatistiksel Modeller ---
    "Logistic Regression": {
        "pros": "Çok hızlı eğitim ve tahmin, düşük işlem maliyeti, katsayıları sayesinde %100 şeffaf ve yorumlanabilir.",
        "cons": "Nitelikler arasındaki doğrusal olmayan karmaşık ilişkileri ve etkileşimleri tek başına yakalayamaz.",
        "best_for": "Açıklanabilirliğin ve hızın kritik olduğu yasal/regülatif karar destek süreçleri."
    },
    "Ridge": {
        "pros": "L2 regülarizasyonu sayesinde çoklu doğrusal bağlantı (multicollinearity) sorununu çözer, kararlı katsayılar üretir.",
        "cons": "Yalnızca doğrusal ilişkileri modeller, değişken seçimi yapmaz.",
        "best_for": "Çok sayıda birbiriyle ilişkili sayısal sütun içeren regresyon analizleri."
    },
    
    # --- Kernel ve Komşuluk Tabanlı Modeller ---
    "SVC": {
        "pros": "Yüksek boyutlu karmaşık uzaylarda etkilidir; net karar sınırları (decision boundary) çizer.",
        "cons": "Büyük veri setlerinde (N > 20.000) eğitim süresi ve bellek tüketimi çok artar; ölçeklendirmeye aşırı duyarlıdır.",
        "best_for": "Küçük/orta boyutlu ve karmaşık sınıflandırma problemleri."
    },
    "SVR": {
        "pros": "Doğrusal olmayan regresyon ilişkilerini marjinal hata toleransı ile başarıyla öğrenir.",
        "cons": "Büyük veri setlerinde ölçeklenmesi zordur; hiperparametre ayarı hassasiyet gerektirir.",
        "best_for": "Küçük boyutlu ama karmaşık dalgalanmalar gösteren regresyon verileri."
    },
    "KNN": {
        "pros": "Ön eğitim gerektirmez (lazy learning); karmaşık ve yerel karar sınırlarını sezgisel olarak öğrenir.",
        "cons": "Tahmin anında tüm veri setini taradığı için yavaştır; aykırı değerlere ve değişken ölçeklerine çok hassastır.",
        "best_for": "Benzerlik tabanlı karar verme ve düşük boyutlu kompakt veri kümeleri."
    },
    
    # --- Anomali Tespiti Modelleri ---
    "Isolation Forest": {
        "pros": "Doğrusal zaman karmaşıklığı O(n) ile çok hızlıdır; veri dağılımı varsayımı yapmaz; yüksek boyutta etkilidir.",
        "cons": "Farklı yoğunluklardaki yerel kümelenme anomalilerini kaçırabilir.",
        "best_for": "Büyük ölçekli kurumsal log, işlem ve sahtekarlık (fraud) anomali taramaları."
    },
    "Local Outlier Factor": {
        "pros": "Yerel veri yoğunluğunu temel alır; farklı yoğunluktaki kümelere ait yerel aykırı değerleri mükemmel yakalar.",
        "cons": "Büyük verilerde komşuluk hesabı maliyetlidir; yeni gelen veriler için anlık çıkarım zordur.",
        "best_for": "Kümelenmiş ve homojen olmayan veri dağılımlarında yerel anomali tespiti."
    },
    "One-Class SVM": {
        "pros": "Normal verinin sınırlarını doğrusal olmayan kernel desteğiyle çok sıkı ve hassas bir şekilde çizer.",
        "cons": "Aykırı değer oranı parametresine (nu) çok duyarlıdır; büyük verilerde yavaşlar.",
        "best_for": "Temiz ve tek tip normal verinin bulunduğu nadir olay tespiti senaryoları."
    },
    "Elliptic Envelope": {
        "pros": "Normal (Gaussian) dağılıma sahip verilerde teorik olarak en tutarlı ve hızlı anomali tespit yöntemidir.",
        "cons": "Verinin çok değişkenli normal dağıldığını varsayar; çarpık veya çok modlu verilerde yanıltıcı olabilir.",
        "best_for": "İyi temizlenmiş ve standart dağılıma uyan finansal/istatistiksel göstergeler."
    }
}

def get_model_guidance(algorithm_name: str) -> dict:
    """Verilen algoritma için güçlü, zayıf yönleri ve en uygun kullanım senaryosunu döner."""
    return MODEL_GUIDANCE.get(algorithm_name, {
        "pros": "Kurumsal makine öğrenmesi standartlarına uygun genel amaçlı algoritma.",
        "cons": "Veri dağılımına ve hiperparametre seçimine göre performansı değişkenlik gösterebilir.",
        "best_for": "Genel kurumsal veri analitiği görevleri."
    })
