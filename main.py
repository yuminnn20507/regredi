import streamlit as st
import pandas as pd
import numpy as np

# -------------------------
# 기존에 선생님이 준 코드
# -------------------------

# 데이터 불러오기
# ...
# all_years 만들기
# ...
# 기존 그래프
# ...


# -------------------------
# 여기부터 도전 문제
# -------------------------

st.subheader("도전 - 직선 대신 곡선을 쓰면")

연평균 = all_years[
    all_years["count"] >= 300
].rename(columns={"mean": "기온"})

학습 = 연평균[
    연평균["연도"] < 2005
]

평가 = 연평균[
    연평균["연도"] >= 2005
]

x = lambda y: (y - 1950) / 100

rows = []

for 차수 in [1, 3, 9]:

    계수 = np.polyfit(
        x(학습["연도"]),
        학습["기온"],
        차수
    )

    예측값 = np.polyval(
        계수,
        x(평가["연도"])
    )

    평가오차 = np.abs(
        예측값 - 평가["기온"]
    ).mean()

    rows.append({
        "곡선": "1차 (직선)" if 차수 == 1 else f"{차수}차",
        "테스트용 데이터의 오차(MAE)": round(평가오차, 2),
        "2050년 예측(℃)": round(
            np.polyval(계수, x(2050)), 1
        )
    })


st.dataframe(
    pd.DataFrame(rows),
    hide_index=True,
    use_container_width=True
)

st.caption(
    f"훈련용 {len(학습)}개 연도 · "
    f"테스트용 {len(평가)}개 연도"
)
