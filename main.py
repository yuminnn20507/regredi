import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# =========================================================
# 페이지 설정
# =========================================================
st.set_page_config(
    page_title="서울 기온 선형회귀 분석",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 연평균 기온 선형회귀 분석")

st.write(
    "서울의 연평균 기온을 이용해 선형회귀 모델을 만들고, "
    "과거 데이터로 학습한 모델이 최근 20년을 얼마나 잘 예측하는지 비교합니다."
)


# =========================================================
# 데이터 주소
# =========================================================
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


# =========================================================
# 데이터 불러오기
# =========================================================
@st.cache_data
def load_data():

    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8"
    )

    # 날짜 변환
    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    # 평균기온 숫자 변환
    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    # 필요한 값이 없는 행 제거
    df = df.dropna(
        subset=["날짜", "평균기온"]
    )

    # 연도 추가
    df["연도"] = df["날짜"].dt.year

    return df


df = load_data()


# =========================================================
# 2025년 이후 제외
# =========================================================
df = df[
    df["연도"] <= 2025
].copy()


# =========================================================
# 연도별 평균기온 / 관측일수
# =========================================================
yearly = (
    df.groupby("연도")
    .agg(
        연평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count")
    )
    .reset_index()
)


# =========================================================
# 관측일수가 300일 미만인 연도 제외
# =========================================================
yearly = yearly[
    yearly["관측일수"] >= 300
].copy()

yearly = yearly.sort_values(
    "연도"
).reset_index(drop=True)


# =========================================================
# 데이터 확인
# =========================================================
if len(yearly) < 2:
    st.error("회귀분석에 사용할 데이터가 부족합니다.")
    st.stop()


# =========================================================
# 학습 / 테스트 기간
# =========================================================
TRAIN_50_START = 1956
TRAIN_50_END = 2005

TRAIN_100_START = 1906
TRAIN_100_END = 2005

TEST_START = 2006
TEST_END = 2025


# =========================================================
# 데이터 분리
# =========================================================

# 최근 50년 학습 데이터
train_50 = yearly[
    (yearly["연도"] >= TRAIN_50_START) &
    (yearly["연도"] <= TRAIN_50_END)
].copy()


# 최근 100년 학습 데이터
train_100 = yearly[
    (yearly["연도"] >= TRAIN_100_START) &
    (yearly["연도"] <= TRAIN_100_END)
].copy()


# 공통 테스트 데이터
test = yearly[
    (yearly["연도"] >= TEST_START) &
    (yearly["연도"] <= TEST_END)
].copy()


# =========================================================
# 데이터 부족 확인
# =========================================================
if len(train_50) < 2:
    st.error("1956~2005년 학습 데이터가 부족합니다.")
    st.stop()

if len(train_100) < 2:
    st.error("1906~2005년 학습 데이터가 부족합니다.")
    st.stop()

if len(test) < 2:
    st.error("2006~2025년 테스트 데이터가 부족합니다.")
    st.stop()


# =========================================================
# X, y 설정
# =========================================================
X_train_50 = train_50[["연도"]]
y_train_50 = train_50["연평균기온"]

X_train_100 = train_100[["연도"]]
y_train_100 = train_100["연평균기온"]

X_test = test[["연도"]]
y_test = test["연평균기온"]


# =========================================================
# 선형회귀 모델 학습
# =========================================================
model_50 = LinearRegression()

model_50.fit(
    X_train_50,
    y_train_50
)


model_100 = LinearRegression()

model_100.fit(
    X_train_100,
    y_train_100
)


# =========================================================
# 테스트 데이터 예측
# =========================================================
pred_50 = model_50.predict(
    X_test
)

pred_100 = model_100.predict(
    X_test
)


# =========================================================
# 테스트 성능 평가 함수
# =========================================================
def evaluate_model(y_true, y_pred):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    return mae, mse, r2


# =========================================================
# 50년 모델 테스트 성능
# =========================================================
mae_50, mse_50, r2_50 = evaluate_model(
    y_test,
    pred_50
)


# =========================================================
# 100년 모델 테스트 성능
# =========================================================
mae_100, mse_100, r2_100 = evaluate_model(
    y_test,
    pred_100
)


# =========================================================
# 기울기
# =========================================================
slope_50 = model_50.coef_[0]
slope_100 = model_100.coef_[0]

# 100년에 몇 도 변하는지
slope_50_100 = slope_50 * 100
slope_100_100 = slope_100 * 100


# =========================================================
# 전체 데이터 모델
# =========================================================
X_all = yearly[["연도"]]
y_all = yearly["연평균기온"]

model_all = LinearRegression()

model_all.fit(
    X_all,
    y_all
)


pred_all = model_all.predict(
    X_all
)


# 전체 데이터 자체에 대한 평가
all_mae = mean_absolute_error(
    y_all,
    pred_all
)

all_mse = mean_squared_error(
    y_all,
    pred_all
)

all_r2 = r2_score(
    y_all,
    pred_all
)

all_slope = model_all.coef_[0]
all_slope_100 = all_slope * 100


# =========================================================
# 1. 데이터 구성
# =========================================================
st.divider()

st.subheader("📚 데이터 구성")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🟦 최근 50년")
    st.write("학습: **1956~2005년**")
    st.write(f"실제 학습 연도: **{len(train_50)}년**")

with col2:
    st.markdown("### 🟩 최근 100년")
    st.write("학습: **1906~2005년**")
    st.write(f"실제 학습 연도: **{len(train_100)}년**")

with col3:
    st.markdown("### 🟥 공통 테스트")
    st.write("테스트: **2006~2025년**")
    st.write(f"실제 테스트 연도: **{len(test)}년**")


st.info(
    "두 모델 모두 2006~2025년 데이터를 학습 과정에서는 사용하지 않습니다. "
    "따라서 같은 테스트 데이터를 이용해 두 모델의 예측 성능을 공정하게 비교합니다."
)


# =========================================================
# 2. 전체 데이터 회귀 결과
# =========================================================
st.divider()

st.subheader("📊 전체 데이터로 만든 회귀모델")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "기울기",
        f"{all_slope_100:+.2f} ℃/100년"
    )

with col2:
    st.metric(
        "MAE",
        f"{all_mae:.3f} ℃"
    )

with col3:
    st.metric(
        "MSE",
        f"{all_mse:.3f}"
    )

with col4:
    st.metric(
        "R²",
        f"{all_r2:.3f}"
    )

st.caption(
    "※ 이 평가는 전체 데이터를 학습한 뒤 같은 전체 데이터에 대해 계산한 값입니다. "
    "따라서 실제 예측 성능 비교에는 아래의 2006~2025년 테스트 결과를 사용하는 것이 적절합니다."
)


# =========================================================
# 3. 기울기 비교
# =========================================================
st.divider()

st.subheader("📈 학습 기간에 따른 기울기 비교")

col1, col2 = st.columns(2)

with col1:

    st.markdown("### 🟦 최근 50년 학습")

    st.metric(
        "100년에 기온이 변하는 정도",
        f"{slope_50_100:+.2f} ℃"
    )

    st.write(
        f"회귀식: "
        f"기온 = {slope_50:.5f} × 연도 "
        f"+ {model_50.intercept_:.2f}"
    )

    st.caption(
        "1956~2005년으로 학습"
    )


with col2:

    st.markdown("### 🟩 최근 100년 학습")

    st.metric(
        "100년에 기온이 변하는 정도",
        f"{slope_100_100:+.2f} ℃"
    )

    st.write(
        f"회귀식: "
        f"기온 = {slope_100:.5f} × 연도 "
        f"+ {model_100.intercept_:.2f}"
    )

    st.caption(
        "1906~2005년으로 학습"
    )


# =========================================================
# 4. 테스트 성능 비교
# =========================================================
st.divider()

st.subheader("🎯 2006~2025년 테스트 성능 비교")

st.write(
    "과거 데이터로 학습한 회귀선이 한 번도 보지 않은 "
    "2006~2025년의 실제 연평균 기온을 얼마나 잘 예측했는지 평가합니다."
)


# -------------------------
# 50년 모델
# -------------------------
st.markdown("### 🟦 최근 50년 학습 → 최근 20년 예측")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "MAE",
        f"{mae_50:.3f} ℃"
    )
    st.caption("낮을수록 좋음")

with col2:
    st.metric(
        "MSE",
        f"{mse_50:.3f}"
    )
    st.caption("낮을수록 좋음")

with col3:
    st.metric(
        "R²",
        f"{r2_50:.3f}"
    )
    st.caption("높을수록 좋음")


# -------------------------
# 100년 모델
# -------------------------
st.markdown("### 🟩 최근 100년 학습 → 최근 20년 예측")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "MAE",
        f"{mae_100:.3f} ℃"
    )
    st.caption("낮을수록 좋음")

with col2:
    st.metric(
        "MSE",
        f"{mse_100:.3f}"
    )
    st.caption("낮을수록 좋음")

with col3:
    st.metric(
        "R²",
        f"{r2_100:.3f}"
    )
    st.caption("높을수록 좋음")


# =========================================================
# 5. 표로 한눈에 비교
# =========================================================
st.markdown("### 📋 두 모델 한눈에 비교")

comparison = pd.DataFrame({
    "모델": [
        "최근 50년 학습",
        "최근 100년 학습"
    ],
    "학습 기간": [
        "1956~2005",
        "1906~2005"
    ],
    "테스트 기간": [
        "2006~2025",
        "2006~2025"
    ],
    "기울기 (℃/100년)": [
        slope_50_100,
        slope_100_100
    ],
    "MAE (℃)": [
        mae_50,
        mae_100
    ],
    "MSE": [
        mse_50,
        mse_100
    ],
    "R²": [
        r2_50,
        r2_100
    ]
})


st.dataframe(
    comparison.style.format({
        "기울기 (℃/100년)": "{:+.3f}",
        "MAE (℃)": "{:.3f}",
        "MSE": "{:.3f}",
        "R²": "{:.3f}"
    }),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 6. 실제값과 예측값 비교 그래프
# =========================================================
st.divider()

st.subheader("📉 실제 기온 vs 회귀모델 예측")

fig = go.Figure()


# 실제값
fig.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["연평균기온"],
        mode="lines+markers",
        name="실제 연평균 기온",
        line=dict(width=3),
        marker=dict(size=7),
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "실제: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# 50년 모델
fig.add_trace(
    go.Scatter(
        x=test["연도"],
        y=pred_50,
        mode="lines",
        name="1956~2005 학습",
        line=dict(
            width=3,
            dash="dash"
        ),
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "50년 모델: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# 100년 모델
fig.add_trace(
    go.Scatter(
        x=test["연도"],
        y=pred_100,
        mode="lines",
        name="1906~2005 학습",
        line=dict(
            width=3,
            dash="dot"
        ),
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "100년 모델: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균 기온 (℃)",
    xaxis=dict(
        tickmode="linear",
        dtick=2
    ),
    height=550,
    hovermode="x unified"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================================================
# 7. 두 회귀선 자체 비교
# =========================================================
st.divider()

st.subheader("📈 50년 회귀선과 100년 회귀선 비교")


plot_years = np.arange(
    1906,
    2026
)


plot_df = pd.DataFrame({
    "연도": plot_years
})


plot_df["50년 회귀선"] = model_50.predict(
    plot_df[["연도"]]
)

plot_df["100년 회귀선"] = model_100.predict(
    plot_df[["연도"]]
)


fig2 = go.Figure()


fig2.add_trace(
    go.Scatter(
        x=plot_df["연도"],
        y=plot_df["50년 회귀선"],
        mode="lines",
        name="1956~2005 학습",
        line=dict(width=3)
    )
)


fig2.add_trace(
    go.Scatter(
        x=plot_df["연도"],
        y=plot_df["100년 회귀선"],
        mode="lines",
        name="1906~2005 학습",
        line=dict(
            width=3,
            dash="dash"
        )
    )
)


fig2.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균 기온",
        marker=dict(size=5)
    )
)


fig2.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균 기온 (℃)",
    height=550,
    hovermode="x unified"
)


st.plotly_chart(
    fig2,
    use_container_width=True
)


# =========================================================
# 8. 자동 해석
# =========================================================
st.divider()

st.subheader("💡 결과 해석")


if mae_50 < mae_100:
    mae_result = "최근 50년 모델"
else:
    mae_result = "최근 100년 모델"


if mse_50 < mse_100:
    mse_result = "최근 50년 모델"
else:
    mse_result = "최근 100년 모델"


if r2_50 > r2_100:
    r2_result = "최근 50년 모델"
else:
    r2_result = "최근 100년 모델"


st.write(
    f"**MAE:** {mae_result}의 평균적인 예측 오차가 더 작습니다."
)

st.write(
    f"**MSE:** {mse_result}이 큰 오차까지 고려했을 때 더 좋은 성능을 보입니다."
)

st.write(
    f"**R²:** {r2_result}이 2006~2025년의 기온 변동을 더 잘 설명합니다."
)


slope_difference = abs(
    slope_50_100 - slope_100_100
)


st.write(
    f"두 모델의 기울기 차이는 "
    f"**{slope_difference:.2f}℃/100년**입니다."
)


if slope_50_100 > slope_100_100:

    st.write(
        "최근 50년으로 학습한 회귀선이 "
        "최근 100년으로 학습한 회귀선보다 더 가파릅니다."
    )

elif slope_50_100 < slope_100_100:

    st.write(
        "최근 100년으로 학습한 회귀선이 "
        "최근 50년으로 학습한 회귀선보다 더 가파릅니다."
    )

else:

    st.write(
        "두 회귀선의 기울기는 거의 같습니다."
    )


st.caption(
    "※ MAE와 MSE는 낮을수록 좋고, R²는 높을수록 좋습니다. "
    "R²가 0보다 작으면 단순히 평균값을 예측하는 것보다도 "
    "테스트 데이터를 잘 설명하지 못했다는 뜻입니다."
)
