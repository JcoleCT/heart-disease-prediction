
import streamlit as st
import pandas as pd
import joblib
import os

# =====================================
# 1. CẤU HÌNH TRANG
# =====================================
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

MODEL_PATH = "models/heart_model.pkl"
DATA_PATH = "data/heart.csv"

# =====================================
# 2. TẢI DỮ LIỆU VÀ MÔ HÌNH
# =====================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


st.title("❤️ HEART DISEASE PREDICTION")
st.subheader(
    "Ứng dụng phân tích dữ liệu và hỗ trợ dự đoán nguy cơ bệnh tim"
)

st.info(
    "Ứng dụng phục vụ mục đích học tập và nghiên cứu. "
    "Kết quả không thay thế chẩn đoán hoặc tư vấn y tế."
)

# Kiểm tra file
if not os.path.exists(MODEL_PATH):
    st.error("Không tìm thấy mô hình models/heart_model.pkl")
    st.stop()

if not os.path.exists(DATA_PATH):
    st.error("Không tìm thấy dữ liệu data/heart.csv")
    st.stop()

model = load_model()
df = load_data()

# =====================================
# 3. MENU ĐIỀU HƯỚNG
# =====================================
menu = st.sidebar.radio(
    "MENU CHÍNH",
    ["Dashboard", "Dự đoán bệnh tim"]
)

# =====================================
# 4. DASHBOARD
# =====================================
if menu == "Dashboard":

    st.header("📊 Tổng quan dữ liệu")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Tổng số mẫu", len(df))

    with col2:
        st.metric("Số đặc trưng", 13)

    with col3:
        st.metric("Số thuật toán đã thử nghiệm", 3)

    st.header("📋 Dữ liệu Heart Disease")

    if st.checkbox("Hiển thị dữ liệu"):
        st.dataframe(df.head(10), use_container_width=True)

    st.header("📈 Phân bố tình trạng bệnh")

    df["target"] = (df["num"] > 0).astype(int)

    distribution = (
        df["target"]
        .value_counts()
        .rename({
            0: "Không có nhãn bệnh",
            1: "Có nhãn bệnh"
        })
    )

    st.bar_chart(distribution)

# =====================================
# 5. CHỨC NĂNG DỰ ĐOÁN
# =====================================
elif menu == "Dự đoán bệnh tim":

    st.header("🩺 Nhập thông tin người dùng")

    st.write(
        "Vui lòng nhập 13 thông số để mô hình thực hiện dự đoán."
    )

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        # Nhóm thông tin 1
        with col1:

            st.subheader("Thông tin cơ bản")

            age = st.number_input(
                "Tuổi",
                min_value=1,
                max_value=120,
                value=50
            )

            sex = st.selectbox(
                "Giới tính",
                options=[1, 0],
                format_func=lambda x:
                    "Nam" if x == 1 else "Nữ"
            )

            cp = st.selectbox(
                "Loại đau ngực (cp)",
                options=[1, 2, 3, 4],
                format_func=lambda x: {
                    1: "1 - Đau thắt ngực điển hình",
                    2: "2 - Đau thắt ngực không điển hình",
                    3: "3 - Đau không phải đau thắt ngực",
                    4: "4 - Không có triệu chứng"
                }[x]
            )

            trestbps = st.number_input(
                "Huyết áp lúc nghỉ (mm Hg)",
                min_value=50,
                max_value=250,
                value=120
            )

            chol = st.number_input(
                "Cholesterol huyết thanh (mg/dl)",
                min_value=50,
                max_value=700,
                value=200
            )

            fbs = st.selectbox(
                "Đường huyết lúc đói > 120 mg/dl",
                options=[0, 1],
                format_func=lambda x:
                    "Không" if x == 0 else "Có"
            )

            restecg = st.selectbox(
                "Kết quả điện tâm đồ lúc nghỉ (restecg)",
                options=[0, 1, 2]
            )

        # Nhóm thông tin 2
        with col2:

            st.subheader("Thông số tim mạch")

            thalach = st.number_input(
                "Nhịp tim tối đa đạt được",
                min_value=50,
                max_value=250,
                value=150
            )

            exang = st.selectbox(
                "Đau thắt ngực khi vận động",
                options=[0, 1],
                format_func=lambda x:
                    "Không" if x == 0 else "Có"
            )

            oldpeak = st.number_input(
                "Oldpeak",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.1
            )

            slope = st.selectbox(
                "Độ dốc đoạn ST khi vận động",
                options=[1, 2, 3]
            )

            ca = st.selectbox(
                "Số mạch máu chính được quan sát (ca)",
                options=[0, 1, 2, 3]
            )

            thal = st.selectbox(
                "Thal",
                options=[3, 6, 7],
                format_func=lambda x: {
                    3: "3 - Bình thường",
                    6: "6 - Khiếm khuyết cố định",
                    7: "7 - Khiếm khuyết có thể hồi phục"
                }[x]
            )

        submitted = st.form_submit_button(
            "🔍 THỰC HIỆN DỰ ĐOÁN",
            use_container_width=True
        )

    # =================================
    # 6. XỬ LÝ KẾT QUẢ DỰ ĐOÁN
    # =================================
    if submitted:

        input_data = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal
        }])

        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        st.divider()
        st.header("📋 KẾT QUẢ DỰ ĐOÁN")

        probability = probabilities[1] * 100

        col1, col2 = st.columns(2)

        with col1:
            if prediction == 1:
                st.error(
                    "Mô hình dự đoán: CÓ NHÃN BỆNH"
                )
            else:
                st.success(
                    "Mô hình dự đoán: KHÔNG CÓ NHÃN BỆNH"
                )

        with col2:
            st.metric(
                "Xác suất mô hình ước tính cho lớp có bệnh",
                f"{probability:.2f}%"
            )

        st.progress(float(probabilities[1]))

        st.warning(
            "Đây là kết quả phân loại của mô hình trên dữ liệu "
            "đầu vào, không phải xác suất mắc bệnh đã được xác "
            "nhận lâm sàng. Không sử dụng kết quả này để tự chẩn "
            "đoán hoặc quyết định điều trị."
        )

        with st.expander("Xem thông tin đầu vào"):
            st.dataframe(
                input_data,
                use_container_width=True
            )
