import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# ==================================================
# 페이지 설정
# ==================================================
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 기온 예측기")
st.write(
    "서울의 연평균 기온 변화를 분석하고 "
    "회귀선을 이용해 연도별 예상 기온을 확인합니다."
)


# ==================================================
# 데이터 불러오기
# ==================================================
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


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


# ==================================================
# 2025년까지의 데이터만 사용
# ==================================================
df = df[df["연도"] <= 2025]


# ==================================================
# 연도별 평균기온 + 관측일수 계산
# ==================================================
yearly = (
    df.groupby("연도")
    .agg(
        연평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count")
    )
    .reset_index()
)


# ==================================================
# 관측일수가 300일 이상인 연도만 사용
# ==================================================
yearly = yearly[
    yearly["관측일수"] >= 300
].copy()


yearly = yearly.sort_values(
    "연도"
).reset_index(drop=True)


# ==================================================
# 데이터가 충분하지 않은 경우
# ==================================================
if len(yearly) < 2:
    st.error(
        "회귀분석을 할 수 있는 데이터가 충분하지 않습니다."
    )
    st.stop()


# ==================================================
# 전체 기간 회귀분석
#
# 독립변수:
# 1908년부터 지난 연수
#
# 1908년 → 0
# 1909년 → 1
# ...
# ==================================================
yearly["경과연수"] = yearly["연도"] - 1908

x = yearly["경과연수"].to_numpy()
y = yearly["연평균기온"].to_numpy()


# 1차 선형 회귀
slope, intercept = np.polyfit(
    x,
    y,
    1
)


# 100년에 몇 ℃ 변하는가?
slope_100 = slope * 100


# 전체 회귀선의 예측값
yearly["회귀예측기온"] = (
    slope * yearly["경과연수"]
    + intercept
)


# 전체 기간 상관계수
correlation = np.corrcoef(
    x,
    y
)[0, 1]


# ==================================================
# 전체 기간
# ==================================================
start_year = int(
    yearly["연도"].min()
)

end_year = int(
    yearly["연도"].max()
)

data_count = len(yearly)


# ==================================================
# 최근 20년 데이터
# ==================================================
recent_end_year = end_year
recent_start_year = recent_end_year - 19


recent = yearly[
    (yearly["연도"] >= recent_start_year)
    & (yearly["연도"] <= recent_end_year)
].copy()


# 최근 20년 데이터가 2개 이상일 때만 회귀
if len(recent) >= 2:

    recent_x = (
        recent["연도"] - recent_start_year
    ).to_numpy()

    recent_y = (
        recent["연평균기온"]
    ).to_numpy()

    recent_slope, recent_intercept = np.polyfit(
        recent_x,
        recent_y,
        1
    )

    # 최근 20년의 100년당 변화량
    recent_slope_100 = recent_slope * 100

else:

    recent_slope = np.nan
    recent_intercept = np.nan
    recent_slope_100 = np.nan


# ==================================================
# 기온 상승 속도 크게 표시
# ==================================================
st.subheader("🔥 기온 상승 속도 비교")

col1, col2 = st.columns(2)


with col1:

    st.markdown("### 전체 기간")

    st.metric(
        "100년에 몇 ℃ 변하는가?",
        f"{slope_100:+.2f} ℃"
    )

    st.caption(
        f"{start_year}~{end_year}년"
    )


with col2:

    st.markdown("### 최근 20년")

    if not np.isnan(recent_slope_100):

        st.metric(
            "100년에 몇 ℃ 변하는가?",
            f"{recent_slope_100:+.2f} ℃"
        )

        st.caption(
            f"{recent_start_year}~{recent_end_year}년"
        )

    else:

        st.metric(
            "100년에 몇 ℃ 변하는가?",
            "계산 불가"
        )


# ==================================================
# 회귀분석 정보
# ==================================================
st.divider()

info_col1, info_col2, info_col3 = st.columns(3)


with info_col1:
    st.metric(
        "회귀에 사용한 연도 수",
        f"{data_count}년"
    )


with info_col2:
    st.metric(
        "시작 연도",
        f"{start_year}년"
    )


with info_col3:
    st.metric(
        "끝 연도",
        f"{end_year}년"
    )


st.write(
    f"**전체 기간 상관계수:** {correlation:.3f}"
)


st.write(
    f"**전체 기간 회귀식:** "
    f"연평균기온 = "
    f"{slope:.4f} × (연도 - 1908) "
    f"+ {intercept:.2f}"
)


# ==================================================
# 산점도 + 전체 기간 회귀선
# ==================================================
st.subheader("📊 연도별 연평균 기온과 회귀선")


fig = go.Figure()


# 실제 관측값
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


# 전체 기간 회귀선
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["회귀예측기온"],
        mode="lines",
        name="전체 기간 회귀선",
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
    height=550,
    hovermode="x unified",
    margin=dict(
        l=40,
        r=40,
        t=30,
        b=40
    ),
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


# ==================================================
# 연도별 예상 기온
# ==================================================
st.divider()

st.subheader("🔮 연도별 예상 기온")


selected_year = st.slider(
    "예측할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)


# 선택한 연도의 경과연수
selected_elapsed = selected_year - 1908


# 회귀식으로 예상 기온 계산
predicted_temp = (
    slope * selected_elapsed
    + intercept
)


# ==================================================
# 예상 기온 크게 표시
# ==================================================
st.markdown(
    f"""
    <div style="
        text-align: center;
        padding: 25px;
        margin: 15px 0 25px 0;
        border-radius: 15px;
        background-color: rgba(128, 128, 128, 0.08);
    ">
        <div style="
            font-size: 24px;
            font-weight: bold;
        ">
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


# ==================================================
# 1900~2100년 전체 예상 회귀선
# ==================================================
prediction_years = np.arange(
    1900,
    2101
)


prediction_temps = (
    slope * (prediction_years - 1908)
    + intercept
)


prediction_fig = go.Figure()


# 회귀선
prediction_fig.add_trace(
    go.Scatter(
        x=prediction_years,
        y=prediction_temps,
        mode="lines",
        name="회귀선",
        line=dict(
            width=3
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
    margin=dict(
        l=40,
        r=40,
        t=30,
        b=40
    )
)


st.plotly_chart(
    prediction_fig,
    use_container_width=True
)


# ==================================================
# 안내
# ==================================================
st.info(
    "※ 2025년까지의 데이터 중 관측일수가 300일 이상인 "
    "연도만 회귀분석에 사용했습니다. "
    "1900~2100년의 예상 기온은 전체 기간의 선형 회귀선을 "
    "연장하여 계산한 값이며 실제 미래 기온을 보장하지 않습니다."
)
