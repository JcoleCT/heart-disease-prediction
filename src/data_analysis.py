import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Đọc dữ liệu
df = pd.read_csv("data/heart.csv")

# 1. Kích thước dữ liệu
print("KÍCH THƯỚC DỮ LIỆU:")
print(df.shape)

# 2. Kiểu dữ liệu
print("\nKIỂU DỮ LIỆU:")
print(df.dtypes)

# 3. Kiểm tra dữ liệu thiếu
print("\nSỐ LƯỢNG DỮ LIỆU THIẾU:")
print(df.isnull().sum())

# 4. Thống kê mô tả
print("\nTHỐNG KÊ MÔ TẢ:")
print(df.describe())

# 5. Kiểm tra số lượng từng nhãn bệnh
print("\nPHÂN BỐ NHÃN NUM:")
print(df["num"].value_counts().sort_index())



# Tạo nhãn nhị phân
df["target"] = (df["num"] > 0).astype(int)

# Cấu hình giao diện biểu đồ
sns.set_theme(style="whitegrid")

# 1. Phân bố độ tuổi
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="age", bins=15, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.show()

# 2. Phân bố bệnh tim
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="target")
plt.title("Heart Disease Distribution")
plt.xlabel("Heart Disease (0: No, 1: Yes)")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.show()

# 3. Cholesterol theo nhóm bệnh
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="target", y="chol")
plt.title("Cholesterol by Heart Disease Group")
plt.xlabel("Heart Disease (0: No, 1: Yes)")
plt.ylabel("Cholesterol")
plt.tight_layout()
plt.show()

# 4. Ma trận tương quan
plt.figure(figsize=(12, 9))
correlation = df.drop(columns=["num"]).corr()
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()