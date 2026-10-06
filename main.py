import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# =========================
# 페이지 설정
# =========================
st.set_page_config(
    page_title="서울 연평균 기온 회귀 분석",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 연평균 기온 선형회귀 분석")
st.markdown(
    "1908년 이후 서울의 연평균 기온 데이터를 이용하여 "
    "과거 50년·100년으로 학습한 회귀모델의 예측 성능을 비교합니다."
)


# =========================
# 데이터 불러오기
# =========================
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    df = df.dropna(subset=["날짜", "평균기온"])

    # 연도 추출
    df["연도"] = df["날짜"].dt.year

    # 2025년까지만 사용
    df = df[df["연도"] <= 2025]

    # 연도별 평균기온과 관측일수 계산
    yearly = (
        df.groupby("연도")
        .agg(
            평균기온=("평균기온", "mean"),
            관측일수=("평균기온", "count")
        )
        .reset_index()
    )

    # 관측일수가 300일 이상인 연도만 사용
    yearly = yearly[yearly["관측일수"] >= 300].copy()

    # 지난번과 동일하게 1908년 기준 경과연수 사용
    yearly["경과연수"] = yearly["연도"] - 1908

    return yearly


yearly = load_data()


# =========================
# 데이터 구간
# =========================

# 50년 학습 데이터
train_50 = yearly[
    (yearly["연도"] >= 1956) &
    (yearly["연도"] <= 2005)
].copy()

# 100년 학습 데이터
train_100 = yearly[
    (yearly["연도"] >= 1906) &
    (yearly["연도"] <= 2005)
].copy()

# 공통 테스트 데이터
test = yearly[
    (yearly["연도"] >= 2006) &
    (yearly["연도"] <= 2025)
].copy()


# =========================
# 선형회귀 함수
# =========================
def linear_regression(x, y):
    """
    y = slope * x + intercept
    형태의 단순 선형회귀
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    slope, intercept = np.polyfit(x, y, 1)

    return slope, intercept


# =========================
# 평가 지표 함수
# =========================
def evaluate_model(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # MAE
    mae = np.mean(np.abs(y_true - y_pred))

    # MSE
    mse = np.mean((y_true - y_pred) ** 2)

    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        r2 = np.nan
    else:
        r2 = 1 - (ss_res / ss_tot)

    return mae, mse, r2


# =========================
# 모델 학습
# =========================

# 50년 모델
slope_50, intercept_50 = linear_regression(
    train_50["경과연수"],
    train_50["평균기온"]
)

# 100년 모델
slope_100, intercept_100 = linear_regression(
    train_100["경과연수"],
    train_100["평균기온"]
)


# =========================
# 테스트 데이터 예측
# =========================

test_x = test["경과연수"].values
test_y = test["평균기온"].values

pred_50 = slope_50 * test_x + intercept_50
pred_100 = slope_100 * test_x + intercept_100


# =========================
# 성능 평가
# =========================

mae_50, mse_50, r2_50 = evaluate_model(
    test_y,
    pred_50
)

mae_100, mse_100, r2_100 = evaluate_model(
    test_y,
    pred_100
)


# =========================
# 전체 데이터 회귀
# =========================

slope_all, intercept_all = linear_regression(
    yearly["경과연수"],
    yearly["평균기온"]
)

all_pred = (
    slope_all * yearly["경과연수"] +
    intercept_all
)

all_mae, all_mse, all_r2 = evaluate_model(
    yearly["평균기온"],
    all_pred
)


# =========================
# 데이터 개수 표시
# =========================

st.subheader("📊 데이터 구분")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "최근 50년 학습",
        f"{len(train_50)}개 연도"
    )
    st.caption("1956 ~ 2005")

with col2:
    st.metric(
        "최근 100년 학습",
        f"{len(train_100)}개 연도"
    )
    st.caption("1906 ~ 2005")

with col3:
    st.metric(
        "공통 테스트",
        f"{len(test)}개 연도"
    )
    st.caption("2006 ~ 2025")


# =========================
# 회귀선 기울기 비교
# =========================

st.subheader("📈 회귀선 기울기 비교")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 최근 50년 학습")
    st.metric(
        "100년당 기온 변화",
        f"{slope_50 * 100:+.2f} ℃"
    )
    st.write(
        f"회귀식: "
        f"y = {slope_50:.4f} × 경과연수 "
        f"+ {intercept_50:.2f}"
    )

with col2:
    st.markdown("### 최근 100년 학습")
    st.metric(
        "100년당 기온 변화",
        f"{slope_100 * 100:+.2f} ℃"
    )
    st.write(
        f"회귀식: "
        f"y = {slope_100:.4f} × 경과연수 "
        f"+ {intercept_100:.2f}"
    )


# =========================
# 테스트 성능 비교
# =========================

st.subheader("🎯 공통 테스트 데이터 예측 성능")

st.caption(
    "테스트 데이터: 2006~2025년의 실제 연평균 기온"
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 최근 50년으로 학습")

    st.metric("MAE", f"{mae_50:.3f} ℃")
    st.metric("MSE", f"{mse_50:.3f}")
    st.metric("R²", f"{r2_50:.3f}")

with col2:
    st.markdown("### 최근 100년으로 학습")

    st.metric("MAE", f"{mae_100:.3f} ℃")
    st.metric("MSE", f"{mse_100:.3f}")
    st.metric("R²", f"{r2_100:.3f}")


# =========================
# 비교표
# =========================

st.subheader("🔎 50년 vs 100년 학습 비교")

comparison = pd.DataFrame({
    "모델": [
        "최근 50년 학습",
        "최근 100년 학습"
    ],
    "학습기간": [
        "1956~2005",
        "1906~2005"
    ],
    "100년당 기온 변화(℃)": [
        slope_50 * 100,
        slope_100 * 100
    ],
    "MAE(℃)": [
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
        "100년당 기온 변화(℃)": "{:+.2f}",
        "MAE(℃)": "{:.3f}",
        "MSE": "{:.3f}",
        "R²": "{:.3f}"
    }),
    use_container_width=True,
    hide_index=True
)


# =========================
# 실제값 vs 예측값
# =========================

st.subheader("📉 테스트 데이터 실제값과 예측값")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["평균기온"],
        mode="lines+markers",
        name="실제 평균기온"
    )
)

fig.add_trace(
    go.Scatter(
        x=test["연도"],
        y=pred_50,
        mode="lines+markers",
        name="50년 학습 예측"
    )
)

fig.add_trace(
    go.Scatter(
        x=test["연도"],
        y=pred_100,
        mode="lines+markers",
        name="100년 학습 예측"
    )
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균 기온 (℃)",
    hovermode="x unified",
    height=500
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================
# 회귀선 비교
# =========================

st.subheader("📈 50년·100년 학습 회귀선 비교")

x_plot = np.linspace(
    yearly["경과연수"].min(),
    yearly["경과연수"].max(),
    300
)

years_plot = x_plot + 1908

y_50 = slope_50 * x_plot + intercept_50
y_100 = slope_100 * x_plot + intercept_100

fig2 = go.Figure()

fig2.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["평균기온"],
        mode="markers",
        name="실제 연평균 기온",
        opacity=0.5
    )
)

fig2.add_trace(
    go.Scatter(
        x=years_plot,
        y=y_50,
        mode="lines",
        name="최근 50년 학습 회귀선"
    )
)

fig2.add_trace(
    go.Scatter(
        x=years_plot,
        y=y_100,
        mode="lines",
        name="최근 100년 학습 회귀선"
    )
)

fig2.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균 기온 (℃)",
    hovermode="x unified",
    height=550
)

st.plotly_chart(
    fig2,
    use_container_width=True
)


# =========================
# 전체 데이터 평가
# =========================

st.subheader("📌 참고: 전체 데이터 회귀모델")

st.write(
    "전체 기간의 데이터를 모두 사용하여 회귀선을 만들고 "
    "같은 데이터에 다시 적용한 결과입니다. "
    "따라서 테스트 성능과 직접 비교하기보다는 참고용으로 봅니다."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "전체 데이터 100년당 변화",
        f"{slope_all * 100:+.2f} ℃"
    )

with col2:
    st.metric(
        "전체 데이터 MAE",
        f"{all_mae:.3f} ℃"
    )

with col3:
    st.metric(
        "전체 데이터 R²",
        f"{all_r2:.3f}"
    )


# =========================
# 자동 해석
# =========================

st.subheader("📝 결과 해석")

if mae_50 < mae_100:
    better_mae = "최근 50년 모델"
else:
    better_mae = "최근 100년 모델"

if r2_50 > r2_100:
    better_r2 = "최근 50년 모델"
else:
    better_r2 = "최근 100년 모델"

st.markdown(
    f"""
- **MAE가 더 작은 모델:** {better_mae}
- **R²가 더 높은 모델:** {better_r2}
- MAE와 MSE는 **작을수록 예측 오차가 작다**는 뜻입니다.
- R²는 **1에 가까울수록 실제 기온의 변화를 잘 설명한다**는 뜻입니다.
- 50년과 100년 모델의 기울기를 비교하면, **최근 기후 변화 추세와 장기간의 기후 변화 추세가 얼마나 다른지** 확인할 수 있습니다.
"""
)
