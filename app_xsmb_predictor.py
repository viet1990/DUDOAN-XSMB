import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta
from collections import Counter
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import numpy as np
import matplotlib.pyplot as plt

DATA_FILE = "xsmb_history.csv"
RESULT_FILE = "ketqua_du_doan.csv"

# -----------------------------
# Hàm xử lý dữ liệu đầu vào
# -----------------------------
def clean_last_2_digits(numbers):
    return [str(n)[-2:].zfill(2) for n in numbers if str(n).strip().isdigit()]

# -----------------------------
# Hàm tính xác suất theo thứ
# -----------------------------
def compute_probabilities_by_weekday(df):
    df["Weekday"] = pd.to_datetime(df.iloc[:, 0], dayfirst=True).dt.day_name()
    weekday_probs = {}
    for day in df["Weekday"].unique():
        sub_df = df[df["Weekday"] == day]
        numbers = clean_last_2_digits(sub_df.iloc[:, 1:].values.flatten())
        counter = Counter(numbers)
        total = sum(counter.values())
        prob_df = pd.DataFrame(counter.items(), columns=["Loto", "Count"])
        prob_df["Probability (%)"] = prob_df["Count"] / total * 100
        weekday_probs[day] = prob_df.sort_values(by="Probability (%)", ascending=False).reset_index(drop=True)
    return weekday_probs

# -----------------------------
# Hàm tính xác suất tổng thể
# -----------------------------
def compute_probabilities(df):
    numbers = clean_last_2_digits(df.iloc[:, 1:].values.flatten())
    counter = Counter(numbers)
    total = sum(counter.values())
    prob_df = pd.DataFrame(counter.items(), columns=["Loto", "Count"])
    prob_df["Probability (%)"] = prob_df["Count"] / total * 100
    return prob_df.sort_values(by="Probability (%)", ascending=False).reset_index(drop=True)

# -----------------------------
# Hàm phân tích chu kỳ xuất hiện
# -----------------------------
def compute_cycle_analysis(df):
    numbers = df.iloc[:, 1:]
    flat = clean_last_2_digits(numbers.values.flatten())
    cycles = {}
    last_seen = {}

    for i, row in enumerate(df.itertuples(index=False)):
        day_numbers = clean_last_2_digits(row[1:])
        for num in day_numbers:
            if num in last_seen:
                diff = i - last_seen[num]
                if num in cycles:
                    cycles[num].append(diff)
                else:
                    cycles[num] = [diff]
            last_seen[num] = i

    result = []
    for num, diffs in cycles.items():
        avg_cycle = round(sum(diffs) / len(diffs), 2)
        result.append((num, avg_cycle))

    cycle_df = pd.DataFrame(result, columns=["Loto", "Avg Cycle"])
    cycle_df = cycle_df.sort_values(by="Avg Cycle")
    return cycle_df

# -----------------------------
# Hàm tính số ngày chưa ra
# -----------------------------
def days_since_last_seen(num, df):
    reversed_df = df.iloc[::-1]
    for i, row in enumerate(reversed_df.itertuples(index=False)):
        day_numbers = clean_last_2_digits(row[1:])
        if num in day_numbers:
            return i
    return len(df)

# -----------------------------
# Chuẩn hóa
# -----------------------------
def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

# -----------------------------
# Hàm đếm số lần cùng xuất hiện của các cặp
# -----------------------------
def count_joint_appearance(df, a, b):
    count = 0
    for row in df.itertuples(index=False):
        nums = clean_last_2_digits(row[1:])
        if a in nums and b in nums:
            count += 1
    return count

# -----------------------------
# Gợi ý xiên thông minh từ top xác suất cao
# -----------------------------
def smart_suggest_xiens(df):
    prob_df = compute_probabilities(df)
    cycle_df = compute_cycle_analysis(df)
    merged = pd.merge(prob_df, cycle_df, on="Loto")
    merged["LastSeen"] = merged["Loto"].apply(lambda x: days_since_last_seen(x, df))
    merged = merged.sort_values(by="Probability (%)", ascending=False).head(20)

    candidates = merged["Loto"].tolist()
    xiens2 = []
    for a, b in combinations(candidates, 2):
        prob_score = (merged.loc[merged["Loto"] == a, "Probability (%)"].values[0] +
                      merged.loc[merged["Loto"] == b, "Probability (%)"].values[0]) / 2
        cycle_score = (1 / merged.loc[merged["Loto"] == a, "Avg Cycle"].values[0] +
                       1 / merged.loc[merged["Loto"] == b, "Avg Cycle"].values[0]) / 2
        joint_score = count_joint_appearance(df[-30:], a, b) / 30
        score = prob_score * 0.5 + cycle_score * 0.3 + joint_score * 0.2
        xiens2.append(((a, b), score))

    xiens2 = sorted(xiens2, key=lambda x: x[1], reverse=True)
    top_xiens2 = [pair for pair, _ in xiens2[:5]]

    top_nums = list(set([num for pair in top_xiens2 for num in pair]))
    xiens3 = list(combinations(top_nums, 3))[:5]
    xiens4 = [top_nums[:4]] if len(top_nums) >= 4 else []

    return top_xiens2, xiens3, xiens4

# -----------------------------
# Gợi ý lô đẹp theo thứ
# -----------------------------
def suggest_top4_by_weekday(df, models):
    today = df.iloc[-1, 0]
    today_weekday = pd.to_datetime(today, dayfirst=True).strftime("%A")
    model = models.get(today_weekday)
    if model is None:
        return [], [], [], []
    prob_df = compute_probabilities(df)
    cycle_df = compute_cycle_analysis(df)
    recent_day = clean_last_2_digits(df.iloc[-1, 1:].dropna().tolist())
    merged = pd.merge(prob_df, cycle_df, on="Loto", how="inner")
    merged["LastSeen"] = merged["Loto"].apply(lambda x: days_since_last_seen(x, df))
    merged = merged[~merged["Loto"].isin(recent_day)]
    merged["NormProb"] = normalize(merged["Probability (%)"])
    merged["NormCycle"] = normalize(1 / merged["Avg Cycle"])
    merged["NormLast"] = normalize(merged["LastSeen"])
    X = merged[["NormProb", "NormCycle", "NormLast"]]
    if len(X) == 0:
        return [], [], [], []
    merged["ModelScore"] = model.predict_proba(X)[:, 1]
    top_los = merged.sort_values(by="ModelScore", ascending=False)["Loto"].head(6).tolist()
    return top_los[:4], list(combinations(top_los[:4], 2)), list(combinations(top_los[:4], 3)), [top_los]

# -----------------------------
# Lưu kết quả gợi ý vào file
# -----------------------------
def save_suggestions_to_file(date_str, top4, xiens2, xiens3, xiens4):
    data = {
        "Ngày": [date_str],
        "Top 4": ["-".join(top4)],
        "Xiên 2": [" | ".join([f"{a}-{b}" for a, b in xiens2])],
        "Xiên 3": [" | ".join([f"{a}-{b}-{c}" for a, b, c in xiens3])],
        "Xiên 4": ["-".join(xiens4[0]) if xiens4 else ""]
    }
    result_df = pd.DataFrame(data)
    if os.path.exists(RESULT_FILE):
        old = pd.read_csv(RESULT_FILE)
        result_df = pd.concat([old, result_df], ignore_index=True)
    result_df.to_csv(RESULT_FILE, index=False)

# -----------------------------
# Giao diện Streamlit
# -----------------------------
st.set_page_config(page_title="Dự đoán XSMB", layout="wide")
st.title("🎯 Dự đoán Lô Tô Miền Bắc - Siêu Chính Xác")

# Tải dữ liệu
uploaded_file = st.file_uploader("📤 Tải file xsmb_history.csv hoặc cập nhật mới:", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.to_csv(DATA_FILE, index=False)
    st.success("✅ Dữ liệu đã được cập nhật và lưu.")
elif os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
else:
    st.error("❌ Không tìm thấy dữ liệu!")
    st.stop()

# Nhập kết quả mới
st.subheader("📝 Nhập kết quả XSMB mới")
with st.form("new_data_form"):
    new_date = st.date_input("Chọn ngày xổ:", value=datetime.today())
    new_numbers = st.text_area("Nhập 27 số (cách nhau bởi dấu cách hoặc dấu phẩy):")
    submitted = st.form_submit_button("Thêm vào dữ liệu")
    if submitted:
        nums = clean_last_2_digits(new_numbers.replace(",", " ").split())
        if len(nums) != 27:
            st.error("⚠️ Cần nhập đúng 27 số!")
        else:
            new_row = pd.DataFrame([[new_date.strftime("%d/%m/%Y")] + nums])
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(DATA_FILE, index=False)
            st.success("✅ Đã thêm kết quả mới vào dữ liệu!")

st.subheader("📅 Kết quả gần nhất")
st.dataframe(df.tail(1), use_container_width=True)

st.info("⏳ Đang huấn luyện mô hình, vui lòng chờ...")
weekday_probs = compute_probabilities_by_weekday(df)
models = {}
for day, probs in weekday_probs.items():
    merged = pd.merge(probs, compute_cycle_analysis(df), on="Loto", how="inner")
    merged["LastSeen"] = merged["Loto"].apply(lambda x: days_since_last_seen(x, df))
    merged["Target"] = merged["Loto"].isin(clean_last_2_digits(df.iloc[-1, 1:].tolist())).astype(int)
    merged["NormProb"] = normalize(merged["Probability (%)"])
    merged["NormCycle"] = normalize(1 / merged["Avg Cycle"])
    merged["NormLast"] = normalize(merged["LastSeen"])
    X = merged[["NormProb", "NormCycle", "NormLast"]]
    y = merged["Target"]

    if len(X) >= 10 and len(set(y)) > 1:
        clf = VotingClassifier(estimators=[
            ("lr", LogisticRegression()),
            ("rf", RandomForestClassifier(n_estimators=100)),
            ("xgb", XGBClassifier(use_label_encoder=False, eval_metric="logloss")),
            ("mlp", MLPClassifier(max_iter=500))
        ], voting="soft")
        clf.fit(X, y)
        models[day] = clf

st.subheader("🌟 Gợi ý Top 4 Lô Đẹp Theo Thứ")
top4, xiens2, xiens3, xiens4 = suggest_top4_by_weekday(df, models)

if not top4:
    st.warning("Không đủ dữ liệu hoặc mô hình chưa sẵn sàng.")
else:
    st.markdown(f"**Top 4:** 🎯 {', '.join(top4)}")
    st.markdown(f"**Xiên 2:** 🔗 {', '.join([f'{a}-{b}' for a, b in xiens2])}")
    st.markdown(f"**Xiên 3:** 🔗 {', '.join([f'{a}-{b}-{c}' for a, b, c in xiens3])}")
    st.markdown(f"**Xiên 4:** 🔗 {xiens4[0] if xiens4 else 'Không đủ số'}")

    today_str = datetime.today().strftime("%d/%m/%Y")
    save_suggestions_to_file(today_str, top4, xiens2, xiens3, xiens4)

st.subheader("📊 Gợi ý xiên thông minh theo thống kê")
x2, x3, x4 = smart_suggest_xiens(df)
st.markdown(f"**Xiên 2:** 🔗 {', '.join([f'{a}-{b}' for a, b in x2])}")
st.markdown(f"**Xiên 3:** 🔗 {', '.join([f'{a}-{b}-{c}' for a, b, c in x3])}")
st.markdown(f"**Xiên 4:** 🔗 {x4[0] if x4 else 'Không đủ số'}")
