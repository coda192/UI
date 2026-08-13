from backend.schemas import CapabilityResponse, DatasetMetadata, ColumnMetadata

CAPABILITIES = CapabilityResponse(
    learning_types=["supervised", "unsupervised"],
    tasks={
        "supervised": ["classification", "regression"],
        "unsupervised": ["anomaly_detection"]
    },
    modes=["local"],
    algorithms={
        "classification": ["Logistic Regression", "Random Forest", "HistGradientBoosting", "SVC", "KNN", "CatBoost", "XGBoost"],
        "regression": ["Ridge", "SVR", "Random Forest", "HistGradientBoosting", "XGBoost"],
        "anomaly_detection": ["Isolation Forest", "Local Outlier Factor", "One-Class SVM", "Elliptic Envelope"]
    },
    model_presets=["fast", "accurate", "interpretable"],
    preprocessing_strategies={
        "missing_value": ["mean", "median", "most_frequent", "drop"],
        "encoding": ["onehot", "label", "target"],
        "scaling": ["standard", "minmax", "robust"]
    },
    validation_options=["holdout", "kfold", "stratified_kfold"],
    tuning_options={
        "sampler": ["tpe", "random", "grid"],
        "pruner": ["median", "hyperband"]
    },
    scoring_options={
        "classification": ["accuracy", "f1", "precision", "recall", "roc_auc"],
        "regression": ["rmse", "mae", "r2"],
        "anomaly_detection": ["f1", "precision", "recall"] # Only applicable if has_ground_truth=True
    },
    evaluation_capabilities=["confusion_matrix", "feature_importance", "residuals", "anomaly_distribution"],
    visualization_capabilities=["roc_curve", "pr_curve", "residual_plot", "feature_importance_plot", "anomaly_score_histogram"]
)

DATASETS = [
    DatasetMetadata(
        id="ds_class_01",
        name="Customer_Churn.csv",
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
