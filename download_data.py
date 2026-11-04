from ucimlrepo import fetch_ucirepo
import pandas as pd
import os

# Tạo thư mục data
os.makedirs("data", exist_ok=True)

# Tải bộ dữ liệu Heart Disease
heart = fetch_ucirepo(id=45)

# Lấy dữ liệu
X = heart.data.features
y = heart.data.targets

# Ghép dữ liệu đầu vào và kết quả
df = pd.concat([X, y], axis=1)

# Lưu dữ liệu
df.to_csv("data/heart.csv", index=False)

print("TẢI DỮ LIỆU THÀNH CÔNG!")
print("\n5 dòng dữ liệu đầu tiên:")
print(df.head())

print("\nThông tin dữ liệu:")
print(df.info())