# 도전 - 직선 대신 곡선을 쓰면

st.subheader("도전 - 직선 대신 곡선을 쓰면")

# 연평균 기온 데이터 준비
연평균 = all_years[
    all_years["count"] >= 300
].rename(columns={"mean": "기온"})

# 2005년 이전 = 학습 데이터
학습 = 연평균[
    연평균["연도"] < 2005
]

# 2005년부터 = 테스트 데이터
평가 = 연평균[
    연평균["연도"] >= 2005
]

# 연도를 작은 숫자로 바꾸기
# 1950년을 0으로 만들고 100년 단위로 줄임
x = lambda y: (y - 1950) / 100


rows = []

for 차수 in [1, 3, 9]:

    # 회귀계수 계산
    계수 = np.polyfit(
        x(학습["연도"]),
        학습["기온"],
        차수
    )

    # 테스트 데이터 예측
    예측값 = np.polyval(
        계수,
        x(평가["연도"])
    )

    # MAE 계산
    평가오차 = np.abs(
        예측값 - 평가["기온"]
    ).mean()

    # 2050년 예측
    예측_2050 = np.polyval(
        계수,
        x(2050)
    )

    rows.append({
        "곡선": f"{차수}차",
        "테스트용 데이터의 오차(MAE)": round(평가오차, 2),
        "2050년 예측(℃)": round(예측_2050, 1)
    })


# 결과 표
st.dataframe(
    pd.DataFrame(rows),
    hide_index=True,
    use_container_width=True
)


# 데이터 개수 표시
st.caption(
    f"훈련용 {len(학습)}개 연도 · "
    f"테스트용 {len(평가)}개 연도 · "
    f"테스트 오차는 평균절대오차(MAE)"
)
