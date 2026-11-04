
import os
import joblib

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from data_preprocessing import preprocess_data


# 1. Chuẩn bị dữ liệu
X_train, X_test, y_train, y_test, preprocessor = preprocess_data()


# 2. Khai báo các mô hình
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )
}


# 3. Thiết lập Cross Validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []
best_model = None
best_score = -1
best_name = ""


# 4. Huấn luyện và đánh giá từng mô hình
for name, classifier in models.items():

    print("\n" + "=" * 55)
    print("MÔ HÌNH:", name)
    print("=" * 55)

    # Pipeline đảm bảo tiền xử lý nằm trong từng fold
    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    # Đánh giá chéo trên tập train
    cv_scores = cross_val_score(
        model_pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="f1"
    )

    mean_cv_f1 = cv_scores.mean()

    print("Cross-validation F1:", round(mean_cv_f1, 4))

    # Huấn luyện trên toàn bộ tập train
    model_pipeline.fit(X_train, y_train)

    # Dự đoán trên tập test
    y_pred = model_pipeline.predict(X_test)
    y_prob = model_pipeline.predict_proba(X_test)[:, 1]

    # Tính các chỉ số
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    print("\nKẾT QUẢ TRÊN TẬP TEST:")
    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1-score :", round(f1, 4))
    print("ROC-AUC  :", round(auc, 4))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    results.append({
        "Model": name,
        "CV F1": mean_cv_f1,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "ROC-AUC": auc
    })

    # Chọn mô hình dựa trên F1 trung bình của Cross Validation
    if mean_cv_f1 > best_score:
        best_score = mean_cv_f1
        best_model = model_pipeline
        best_name = name


# 5. In bảng so sánh
print("\n" + "=" * 70)
print("BẢNG SO SÁNH CÁC MÔ HÌNH")
print("=" * 70)

for result in results:
    print(result)


# 6. Lưu mô hình tốt nhất
os.makedirs("models", exist_ok=True)

joblib.dump(best_model, "models/heart_model.pkl")

print("\nMÔ HÌNH ĐƯỢC CHỌN:", best_name)
print("Cross-validation F1:", round(best_score, 4))
print("Đã lưu mô hình tại: models/heart_model.pkl")
