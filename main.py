# --------------------------------------------------
# 전체 기간 회귀분석
# --------------------------------------------------
yearly["경과연수"] = yearly["연도"] - 1908

x = yearly["경과연수"].to_numpy()
y = yearly["연평균기온"].to_numpy()

# 1차 선형 회귀
slope, intercept = np.polyfit(x, y, 1)

# 100년에 몇 ℃ 변하는가?
slope_100 = slope * 100

# 회귀선 예측값
yearly["회귀예측기온"] = (
    slope * yearly["경과연수"] + intercept
)

# 상관계수
correlation = np.corrcoef(x, y)[0, 1]


# --------------------------------------------------
# 최근 20년 회귀분석
# --------------------------------------------------
recent_end_year = int(yearly["연도"].max())
recent_start_year = recent_end_year - 19

recent = yearly[
    (yearly["연도"] >= recent_start_year) &
    (yearly["연도"] <= recent_end_year)
].copy()

recent_x = recent["연도"].to_numpy() - recent_start_year
recent_y = recent["연평균기온"].to_numpy()

recent_slope, recent_intercept = np.polyfit(
    recent_x,
    recent_y,
    1
)

# 최근 20년의 100년당 변화량
recent_slope_100 = recent_slope * 100


# --------------------------------------------------
# 회귀선의 시작/끝 연도
# --------------------------------------------------
start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())
data_count = len(yearly)


# --------------------------------------------------
# 화면에 회귀 정보 표시
# --------------------------------------------------
st.subheader("📈 기온 상승 속도 비교")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 전체 기간")
    st.metric(
        "100년에 몇 ℃ 변하는가?",
        f"{slope_100:+.2f} ℃"
    )
    st.caption(
        f"{start_year}~{end_year}년, "
        f"회귀에 사용한 {data_count}개 연도"
    )

with col2:
    st.markdown("### 최근 20년")
    st.metric(
        "100년에 몇 ℃ 변하는가?",
        f"{recent_slope_100:+.2f} ℃"
    )
    st.caption(
        f"{recent_start_year}~{recent_end_year}년"
    )


st.divider()

# 상관계수
st.write(
    f"**전체 기간 상관계수:** {correlation:.3f}"
)

st.write(
    f"**전체 기간 회귀식:** "
    f"연평균기온 = {slope:.4f} × (연도 - 1908) + {intercept:.2f}"
)
