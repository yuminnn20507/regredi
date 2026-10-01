import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")
st.write(
    "서울의 연평균 기온 데이터를 이용해 추세를 확인하고 "
    "회귀선을 바탕으로 연도별 예상 기온을 확인해 보세요."
)


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    # 날짜와 평균기온을 숫자/날짜 형태로 변환
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    # 날짜 또는 평균기온이 없는 행 제거
    df = df.dropna(subset=["날짜", "평균기온"])

    # 연도 추출
    df["연도"] = df["날짜"].dt.year

    return df


df = load_data()


# --------------------------------------------------
# 수업 기준 기간 적용
# 1. 2025년 이후 데이터 제외
# 2. 관측일이 300일 미만인 연도 제외
# --------------------------------------------------
df = df[df["연도"] <= 2025]

yearly = (
    df.groupby("연도")
    .agg(
        연평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count")
    )
    .reset_index()
)

# 관측일이 300일 이상인 해만 사용
yearly = yearly[yearly["관측일수"] >= 300].copy()

yearly = yearly.sort_values("연도").reset_index(drop=True)


# --------------------------------------------------
# 회귀분석
# 독립변수: 1908년부터 지난 연수
# 예:
# 1908년 → 0
# 1909년 → 1
# 2025년 → 117
# --------------------------------------------------
yearly["경과연수"] = yearly["연도"] - 1908

x = yearly["경과연수"].to_numpy()
y = yearly["연평균기온"].to_numpy()

# 1차 선형 회귀
slope, intercept = np.polyfit(x, y, 1)

# 회귀선 예측값
yearly["회귀예측기온"] = (
    slope * yearly["경과연수"] + intercept
)

# 상관계수
correlation = np.corrcoef(x, y)[0, 1]


# --------------------------------------------------
# 회귀선의 시작/끝 연도
# --------------------------------------------------
start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())
data_count = len(yearly)


# --------------------------------------------------
# 회귀식
# --------------------------------------------------
st.subheader("📊 서울 연평균 기온과 회귀선")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "회귀에 사용한 연도 수",
        f"{data_count}년"
    )

with col2:
    st.metric(
        "시작 연도",
        f"{start_year}년"
    )

with col3:
    st.metric(
        "끝 연도",
        f"{end_year}년"
    )


st.write(
    f"**상관계수:** {correlation:.3f}"
)

st.write(
    f"**회귀식:** "
    f"연평균기온 = {slope:.4f} × (연도 - 1908) + {intercept:.2f}"
)


# --------------------------------------------------
# 산점도 + 회귀선
# --------------------------------------------------
fig = go.Figure()

# 실제 연평균 기온
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균 기온",
        marker=dict(
            size=7
        ),
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "연평균 기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)

# 회귀선
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["회귀예측기온"],
        mode="lines",
        name="회귀선",
        line=dict(
            width=3
        ),
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "회귀 예상 기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균 기온 (℃)",
    xaxis=dict(
        tickmode="linear",
        dtick=10
    ),
    hovermode="x unified",
    height=550,
    margin=dict(l=40, r=40, t=30, b=40),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# --------------------------------------------------
# 연도 선택
# --------------------------------------------------
st.divider()

st.subheader("🔮 연도별 예상 기온")

selected_year = st.slider(
    "예측할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)


# 회귀식으로 선택 연도의 예상 기온 계산
selected_elapsed = selected_year - 1908

predicted_temp = (
    slope * selected_elapsed + intercept
)


# --------------------------------------------------
# 예상 기온 크게 표시
# --------------------------------------------------
st.markdown(
    f"""
    <div style="
        text-align: center;
        padding: 25px;
        margin: 15px 0 25px 0;
        border-radius: 15px;
        background-color: rgba(128, 128, 128, 0.08);
    ">
        <div style="font-size: 24px; font-weight: bold;">
            {selected_year}년 예상 연평균 기온
        </div>
        <div style="
            font-size: 55px;
            font-weight: bold;
            margin-top: 10px;
        ">
            {predicted_temp:.2f}℃
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# 선택한 연도 위치를 회귀선에 표시
# --------------------------------------------------
prediction_years = np.arange(1900, 2101)

prediction_temps = (
    slope * (prediction_years - 1908) + intercept
)

prediction_fig = go.Figure()

# 전체 회귀선
prediction_fig.add_trace(
    go.Scatter(
        x=prediction_years,
        y=prediction_temps,
        mode="lines",
        name="회귀선",
        line=dict(width=3),
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "예상 기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)

# 선택한 연도
prediction_fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers",
        name=f"{selected_year}년",
        marker=dict(
            size=15
        ),
        hovertemplate=(
            f"<b>{selected_year}년</b><br>"
            f"예상 기온: {predicted_temp:.2f}℃"
            "<extra></extra>"
        )
    )
)

prediction_fig.update_layout(
    xaxis_title="연도",
    yaxis_title="예상 연평균 기온 (℃)",
    xaxis=dict(
        range=[1900, 2100],
        tickmode="linear",
        dtick=10
    ),
    height=450,
    hovermode="x unified",
    margin=dict(l=40, r=40, t=30, b=40)
)

st.plotly_chart(
    prediction_fig,
    use_container_width=True
)


# --------------------------------------------------
# 참고 정보
# --------------------------------------------------
st.info(
    "※ 예상 기온은 2025년까지의 관측 자료 중 "
    "관측일수가 300일 이상인 연도만 사용하여 만든 "
    "단순 선형 회귀 결과입니다. 실제 미래 기온을 보장하는 값은 아닙니다."
)
