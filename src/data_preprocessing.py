
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def preprocess_data():

    # 1. Đọc dữ liệu từ file CSV
    df = pd.read_csv("data/heart.csv")

    # 2. Chuyển nhãn num thành phân loại nhị phân
    # num = 0: Không ghi nhận bệnh
    # num > 0: Có ghi nhận bệnh
    df["target"] = (df["num"] > 0).astype(int)

    # 3. Tách đặc trưng đầu vào (X) và nhãn (y)
    X = df.drop(columns=["num", "target"])
    y = df["target"]

    # 4. Khai báo các cột dữ liệu số
    numerical_features = [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak"
    ]

    # 5. Khai báo các cột dữ liệu phân loại
    categorical_features = [
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal"
    ]

    # 6. Chia dữ liệu thành tập train và test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 7. Pipeline xử lý dữ liệu số
    numerical_pipeline = Pipeline(steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    # 8. Pipeline xử lý dữ liệu phân loại
    categorical_pipeline = Pipeline(steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    # 9. Kết hợp các bước tiền xử lý
    preprocessor = ColumnTransformer(transformers=[
        (
            "num",
            numerical_pipeline,
            numerical_features
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_features
        )
    ])

    # 10. Trả về dữ liệu gốc chưa tiền xử lý
    # Pipeline trong train_model.py sẽ xử lý dữ liệu
    # trong từng fold Cross Validation và khi huấn luyện.

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    )


# 11. Kiểm tra khi chạy trực tiếp file này
if __name__ == "__main__":

    X_train, X_test, y_train, y_test, preprocessor = preprocess_data()

    print("TIỀN XỬ LÝ DỮ LIỆU THÀNH CÔNG!")

    print("\nKÍCH THƯỚC DỮ LIỆU:")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)

    print("\nPHÂN BỐ NHÃN TRONG TẬP TRAIN:")
    print(y_train.value_counts())

    print("\nPHÂN BỐ NHÃN TRONG TẬP TEST:")
    print(y_test.value_counts())

    print("\nĐÃ SẴN SÀNG CHO MACHINE LEARNING!")
