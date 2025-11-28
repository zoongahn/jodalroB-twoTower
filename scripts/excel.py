import pandas as pd
from pathlib import Path

# 1) 엑셀 로드
xls_path = "/data/dev/jodalroB-twoTower/meta/학습용데이터 명세서 0819.xlsx"  # 위치에 맞게 수정
df = pd.read_excel(xls_path, sheet_name="Sheet 1")

# 2) 헤더 정리
df.columns = [
    "table", "column", "type", "use", "null_strategy",
    "is_categorical", "num_categories", "num_nulls",
    "length", "pk", "nn", "description_ko", "note"
]

df = df[df["table"] != "테이블명"].reset_index(drop=True)

# 3) notice / company 만 사용 + 한글 설명 있는 것만
map_df = df[df["table"].isin(["notice", "company"])].copy()
map_df = map_df[~map_df["description_ko"].isna()]

# 4) 컬럼 이름을 feature_importance_v2 포맷에 맞게 맞추기
map_df = map_df.rename(columns={
    "column": "feature",
    "description_ko": "display_name_ko"
})

# 5) 필요한 최소 컬럼만 저장
map_df = map_df[["table", "feature", "display_name_ko"]]

# 6) CSV로 저장
out_path = Path("meta/feature_name_map.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)
map_df.to_csv(out_path, index=False, encoding="utf-8-sig")

print(f"✅ 저장 완료: {out_path}")
print(map_df.head(10))
