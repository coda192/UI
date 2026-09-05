from backend.schemas import CapabilityResponse, AlgorithmMetadata, DatasetMetadata, ColumnMetadata

ALGORITHM_METADATA = {
    # --- Classification ---
    "logreg": AlgorithmMetadata(
        display_name="Lojistik Regresyon (Logistic Regression)",
        description="Doğrusal karar sınırları kullanan, istatistiksel olasılık temelli temel sınıflandırma algoritması.",
        strengths=[
            "Hızlı eğitim ve düşük çıkarım gecikmesi",
            "Yüksek şeffaflık ve katsayı yorumlanabilirliği",
            "Düşük bellek ve işlem kaynağı gereksinimi"
        ],
        limitations=[
            "Doğrusal olmayan karmaşık ilişkileri tek başına modelleyemez",
            "Çoklu doğrusal bağlantıya ve aykırı değerlere duyarlıdır"
        ],
        best_for=[
            "Açıklanabilirliğin yasal olarak zorunlu olduğu regülatif süreçler",
            "Doğrusal ayrılabilir temel sınıflandırma görevleri"
        ]
    ),
    "random_forest_c": AlgorithmMetadata(
        display_name="Random Forest Sınıflandırıcı (Random Forest Classifier)",
        description="Bagging yöntemiyle çok sayıda karar ağacının oylamasını birleştiren topluluk (ensemble) modeli.",
        strengths=[
            "Aşırı öğrenmeye (overfitting) karşı dirençli",
            "Aykırı değerlere ve gürültüye dayanıklı",
            "Nitelik önem düzeylerini (Feature Importance) net sunar"
        ],
        limitations=[
            "Büyük veri kümelerinde bellek kullanımı artabilir",
            "Eğitim kümesi dışındaki uç değerlere ekstrapolasyon yapamaz"
        ],
        best_for=[
            "Dengeli ve güvenilir genel amaçlı tablosal sınıflandırma",
            "Doğrusal olmayan karmaşık etkileşimler içeren veri setleri"
        ]
    ),
    "hgb_c": AlgorithmMetadata(
        display_name="Histogram Tabanlı Gradyan Artırma Sınıflandırıcı (HistGradientBoosting Classifier)",
        description="Sayısal değişkenleri histogram kutularına bölerek hızlı ağaç oluşturmayı sağlayan gradyan artırma modeli.",
        strengths=[
            "Büyük ölçekli veri setlerinde olağanüstü hızlı eğitim",
            "Eksik değerleri (NaN) yerleşik işleyebilme",
            "Bellek verimliliği yüksek"
        ],
        limitations=[
            "Çok küçük veri setlerinde aşırı uyum (overfit) gösterebilir"
        ],
        best_for=[
            "10.000+ satırlı büyük tablosal veri setleri",
            "Eksik değer içeren hızlı prototipleme ve iterasyon süreçleri"
        ]
    ),
    "svc": AlgorithmMetadata(
        display_name="Destek Vektör Sınıflandırıcı (Support Vector Classifier - SVC)",
        description="Sınıflar arasındaki geometrik marjini maksimize eden hiper-düzlemler belirleyen optimizasyon tabanlı algoritma.",
        strengths=[
            "Yüksek boyutlu ve karmaşık uzaylarda etkilidir",
            "Net ve güçlü karar sınırları oluşturur"
        ],
        limitations=[
            "Büyük veri setlerinde eğitim süresi ve bellek tüketimi çok yüksektir",
            "Ölçeklendirmeye (scaling) aşırı duyarlıdır"
        ],
        best_for=[
            "Küçük ve orta ölçekli karmaşık sınıflandırma problemleri",
            "Boyut sayısı yüksek kompakt veri kümeleri"
        ]
    ),
    "knn": AlgorithmMetadata(
        display_name="K-En Yakın Komşu (K-Nearest Neighbors - KNN)",
        description="Önceden bir model eğitmeden, tahmin anında en yakın komşuların çoğunluk sınıfına göre karar veren tembel öğrenme (lazy learning) algoritması.",
        strengths=[
            "Açık bir eğitim evresi gerektirmez",
            "Doğrusal olmayan yerel karar sınırlarını kolayca yakalar"
        ],
        limitations=[
            "Büyük veri kümelerinde tahmin (çıkarım) süresi çok yavaştır",
            "Değişken ölçeklerine ve aykırı değerlere hassastır"
        ],
        best_for=[
            "Düşük boyutlu ve kompakt veri setleri",
            "Benzerlik tabanlı karar destek durumları"
        ]
    ),
    "catboost": AlgorithmMetadata(
        display_name="CatBoost Sınıflandırıcı (CatBoost Classifier)",
        description="Kategorik özellikleri hedef istatistikleri ve simetrik karar ağaçları (oblivious trees) ile işleyen gelişmiş gradyan artırma algoritması.",
        strengths=[
            "Kategorik değişkenlerde üstün başarım ve yerleşik kodlama",
            "Varsayılan hiperparametrelerle yüksek doğruluk",
            "Aşırı öğrenmeye karşı yüksek direnç"
        ],
        limitations=[
            "Yüksek boyutlu seyrek (sparse) matrislerde eğitim süresi uzayabilir"
        ],
        best_for=[
            "Zengin kategorik sütunlar içeren kurumsal tablosal veri setleri",
            "Üretim ortamı sınıflandırma boru hatları"
        ]
    ),
    "xgb_c": AlgorithmMetadata(
        display_name="XGBoost Sınıflandırıcı (XGBoost Classifier)",
        description="İkinci derece Taylor serisi genişlemesi ve L1/L2 regülarizasyonu kullanan optimize edilmiş gradyan artırma algoritması.",
        strengths=[
            "Tablosal sınıflandırmada endüstri standardı başarım",
            "Gelişmiş regülarizasyon ile kontrollü karmaşıklık",
            "Yüksek hız ve paralelleştirme desteği"
        ],
        limitations=[
            "Hiperparametre optimizasyonuna duyarlıdır",
            "Kategorik veriler için ön işleme/kodlama gerektirebilir"
        ],
        best_for=[
            "Yüksek doğruluk hedeflenen yarışma ve üretim seviyesi modeller",
            "Performans kritik tablosal analizler"
        ]
    ),
    
    # --- Regression ---
    "ridge": AlgorithmMetadata(
        display_name="Ridge Regresyon (L2 Regularized Linear Regression)",
        description="Katsayıların büyüklüğüne L2 ceza terimi ekleyerek çoklu doğrusal bağlantıyı engelleyen doğrusal regresyon modeli.",
        strengths=[
            "Çoklu doğrusal bağlantı (multicollinearity) durumunda kararlıdır",
            "Analitik çözümü hızlıdır ve hesaplama maliyeti düşüktür",
            "Katsayılar doğrudan yorumlanabilir"
        ],
        limitations=[
            "Yalnızca doğrusal ilişkileri modelleyebilir",
            "Değişken seçimi yapmaz (katsayıları sıfırlamaz)"
        ],
        best_for=[
            "Çok sayıda ilişkili sayısal sütun içeren regresyon analizleri",
            "Doğrusal temel çizgisi (baseline) oluşturma"
        ]
    ),
    "svr": AlgorithmMetadata(
        display_name="Destek Vektör Regresyon (Support Vector Regressor - SVR)",
        description="Belirlenen bir epsilon marjini içindeki hataları tolere ederek doğrusal olmayan kernel dönüşümleri uygulayan regresyon algoritması.",
        strengths=[
            "Doğrusal olmayan karmaşık ilişkileri ve dalgalanmaları iyi öğrenir",
            "Marjinal hata toleransı ile aykırı değerlere dirençlidir"
        ],
        limitations=[
            "Büyük veri kümelerinde ölçeklenmesi zordur",
            "Hiperparametre ayarına duyarlıdır"
        ],
        best_for=[
            "Küçük/orta boyutlu ve karmaşık trendler içeren regresyon problemleri"
        ]
    ),
    "random_forest_r": AlgorithmMetadata(
        display_name="Random Forest Regresyon (Random Forest Regressor)",
        description="Çok sayıda karar ağacının sayısal tahmin ortalamasını alan topluluk (ensemble) regresyon modeli.",
        strengths=[
            "Doğrusal olmayan etkileşimleri yakalar",
            "Aykırı değerlere karşı dayanıklıdır",
            "Aşırı uyum riski düşüktür"
        ],
        limitations=[
            "Eğitim kümesindeki hedef değişken aralığının dışındaki değerleri tahmin edemez"
        ],
        best_for=[
            "Kararlı ve dengeli genel amaçlı tablosal regresyon tahminleri"
        ]
    ),
    "hgb_r": AlgorithmMetadata(
        display_name="Histogram Tabanlı Gradyan Artırma Regresyon (HistGradientBoosting Regressor)",
        description="Sayısal değişkenleri histogram kutularına bölerek regresyon ağaçları eğiten ölçeklenebilir gradyan artırma algoritması.",
        strengths=[
            "Büyük regresyon verilerinde çok hızlı eğitim",
            "Yerleşik eksik değer (NaN) desteği",
            "Düşük bellek tüketimi"
        ],
        limitations=[
            "Çok küçük veri setlerinde basit regresyonlara göre aşırı uyum gösterebilir"
        ],
        best_for=[
            "Geniş hacimli tablosal regresyon veri setleri"
        ]
    ),
    "xgb": AlgorithmMetadata(
        display_name="XGBoost Regresyon (XGBoost Regressor)",
        description="Kayıp fonksiyonunun gradyan ve hessian değerlerini kullanarak artık değerleri optimize eden gradyan artırma regresyon modeli.",
        strengths=[
            "Regresyonda yüksek kestirim gücü ve düşük hata oranları",
            "L1/L2 regülarizasyon ile aşırı uyum kontrolü",
            "Hızlı optimizasyon"
        ],
        limitations=[
            "Hiperparametre ayarlarına hassastır"
        ],
        best_for=[
            "Yüksek hassasiyet gerektiren sayısal ve finansal tahmin modelleri"
        ]
    ),
    
    # --- Anomaly Detection ---
    "isolation_forest": AlgorithmMetadata(
        display_name="Isolation Forest (Aykırı Değer Yalıtımı)",
        description="Veri noktalarını rastgele bölmelerle izole eden; anomalilerin daha az bölünmeyle izole olduğu prensibine dayanan algoritma.",
        strengths=[
            "Doğrusal O(n) zaman karmaşıklığı ile çok hızlıdır",
            "Veri dağılımı hakkında varsayım yapmaz",
            "Yüksek boyutlu verilerde etkilidir"
        ],
        limitations=[
            "Farklı yoğunluklardaki yerel kümelenme anomalilerini kaçırabilir"
        ],
        best_for=[
            "Büyük ölçekli log, işlem ve sahtekarlık (fraud) anomali taramaları"
        ]
    ),
    "lof": AlgorithmMetadata(
        display_name="Local Outlier Factor (LOF - Yerel Aykırı Değer Faktörü)",
        description="Bir noktanın yerel yoğunluğunu komşularının yoğunluğu ile karşılaştırarak yoğunluk farkına dayalı anomali skoru hesaplayan yöntem.",
        strengths=[
            "Farklı yoğunluktaki kümelere sahip veri setlerinde yerel anomalileri yakalar"
        ],
        limitations=[
            "Büyük veri kümelerinde komşuluk hesabı yüksek bellek ve zaman gerektirir",
            "Yeni gelen veriler için anlık çıkarım zordur"
        ],
        best_for=[
            "Homojen olmayan kümelenmiş veri dağılımlarında yerel anomali tespiti"
        ]
    ),
    "one_class_svm": AlgorithmMetadata(
        display_name="One-Class SVM (Tek Sınıflı Destek Vektör Makinesi)",
        description="Normal veriyi orijinden ayıran en sıkı karar sınırını kernel fonksiyonları kullanarak çizen denetimsiz algoritma.",
        strengths=[
            "Doğrusal olmayan sınırları yüksek boyutta hassas bir şekilde öğrenir"
        ],
        limitations=[
            "Aykırı değer oranı ve kernel parametrelerine çok duyarlıdır",
            "Büyük verilerde yavaşlar"
        ],
        best_for=[
            "Temiz normal verinin bulunduğu nadir arıza ve olay tespiti"
        ]
    ),
    "elliptic_envelope": AlgorithmMetadata(
        display_name="Elliptic Envelope (Eliptik Zarf / Gauss Kovaryans)",
        description="Verinin çok değişkenli normal (Gaussian) dağıldığı varsayımıyla sağlam bir kovaryans tahmini yaparak aykırı değerleri belirleyen yöntem.",
        strengths=[
            "Normal dağılıma uyan verilerde teorik olarak en tutarlı ve hızlı yöntemdir"
        ],
        limitations=[
            "Verinin çok değişkenli normal dağıldığını varsayar; çarpık veya çok modlu verilerde yanıltıcı olabilir"
        ],
        best_for=[
            "Standart istatistiksel dağılıma uyan temiz finansal ve sensör göstergeleri"
        ]
    )
}

CAPABILITIES = CapabilityResponse(
    learning_types=["supervised", "unsupervised"],
    tasks={
        "supervised": ["classification", "regression"],
        "unsupervised": ["anomaly_detection"]
    },
    modes=["train", "tune"],
    algorithms={
        "classification": ["logreg", "random_forest_c", "hgb_c", "svc", "knn", "catboost", "xgb_c"],
        "regression": ["ridge", "svr", "random_forest_r", "xgb", "hgb_r"],
        "anomaly_detection": ["isolation_forest", "lof", "one_class_svm", "elliptic_envelope"]
    },
    model_presets=["baseline", "fast", "strong", "custom"],
    preprocessing_strategies={
        "missing_value": ["mean", "median", "most_frequent", "constant", "drop"],
        "encoding": ["onehot", "frequency", "hashing"],
        "scaling": ["standard", "minmax", "robust"]
    },
    validation_options=["holdout", "kfold", "stratified_kfold"],
    tuning_options={
        "sampler": ["tpe", "random"],
        "pruner": ["none", "median", "sha"],
        "space_preset": ["baseline", "deep"],
        "aggregation": ["mean", "median"]
    },
    scoring_options={
        "classification": ["f1_macro", "accuracy", "balanced_accuracy"],
        "regression": ["neg_mean_squared_error", "r2", "rmse", "mae"],
        "anomaly_detection": ["f1_score", "accuracy", "anomaly_count", "anomaly_ratio"]
    },
    evaluation_capabilities=["confusion_matrix", "feature_importance", "residuals", "anomaly_distribution"],
    visualization_capabilities=["roc_curve", "pr_curve", "residual_plot", "feature_importance_plot", "anomaly_score_histogram"],
    algorithm_metadata=ALGORITHM_METADATA
)

DATASETS = [
    DatasetMetadata(
        id="ds_class_01",
        name="Customer_Churn.csv",
        display_name="Müşteri Kayıp Analizi (Customer Churn)",
        description="Telekomünikasyon sektöründe müşteri demografisi, fatura tutarları ve abonelik süreleri üzerinden kayıp (churn) tahmin veri seti.",
        row_count=10000,
        column_count=12,
        columns=[
            ColumnMetadata(name="CustomerID", dtype="string", missing_count=0),
            ColumnMetadata(name="Age", dtype="int", missing_count=0),
            ColumnMetadata(name="MonthlyCharge", dtype="float", missing_count=15),
            ColumnMetadata(name="Churn", dtype="int", missing_count=0)
        ],
        target="Churn",
        identifier_columns=["CustomerID"],
        compatible_tasks=["classification"]
    ),
    DatasetMetadata(
        id="ds_reg_01",
        name="House_Prices.csv",
        display_name="Konut Fiyat Tahmini (House Prices)",
        description="Konutların yapısal özellikleri, konumları ve arsa büyüklüklerine göre satış fiyatlarını tahmin etmeye yönelik regresyon veri seti.",
        row_count=1460,
        column_count=80,
        columns=[
            ColumnMetadata(name="Id", dtype="int", missing_count=0),
            ColumnMetadata(name="LotArea", dtype="int", missing_count=0),
            ColumnMetadata(name="SalePrice", dtype="float", missing_count=0)
        ],
        target="SalePrice",
        identifier_columns=["Id"],
        compatible_tasks=["regression"]
    ),
    DatasetMetadata(
        id="ds_anom_labeled_01",
        name="CreditCard_Fraud_Labeled.csv",
        display_name="Etiketli Kredi Kartı Dolandırıcılığı (Credit Card Fraud)",
        description="Kredi kartı işlem tutarları ve zaman serisi nitelikleri üzerinden dolandırıcılık tespiti için hazırlanmış etiketli (ground-truth) anomali veri seti.",
        row_count=284807,
        column_count=31,
        columns=[
            ColumnMetadata(name="Time", dtype="float", missing_count=0),
            ColumnMetadata(name="Amount", dtype="float", missing_count=0),
            ColumnMetadata(name="Class", dtype="int", missing_count=0)
        ],
        target="Class",
        identifier_columns=[],
        compatible_tasks=["anomaly_detection"]
    ),
    DatasetMetadata(
        id="ds_anom_unlabeled_01",
        name="Server_Logs_Unlabeled.csv",
        display_name="Etiketsiz Sunucu Metrik Logları (Server Logs)",
        description="Sunucu CPU ve bellek kullanım metriklerini içeren, gerçek etiket bulunmayan denetimsiz anomali tespiti veri seti.",
        row_count=50000,
        column_count=15,
        columns=[
            ColumnMetadata(name="LogID", dtype="string", missing_count=0),
            ColumnMetadata(name="CPU_Usage", dtype="float", missing_count=0),
            ColumnMetadata(name="Mem_Usage", dtype="float", missing_count=0)
        ],
        target=None,
        identifier_columns=["LogID"],
        compatible_tasks=["anomaly_detection"]
    )
]
