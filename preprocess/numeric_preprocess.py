# numeric_preprocess.py
import pandas as pd
import numpy as np
import json
import pickle
from typing import Dict, Any, Optional, List
from pathlib import Path
import os

TABLE_PK_MAP = {
    'notice': ['bidntceno', 'bidntceord'],  # 공고 테이블 PK
    'company': ['bizno']  # 업체 테이블 PK
}


class NumericPreprocessor:
    """수치형 데이터 전처리 클래스"""

    def __init__(self, config: Dict[str, Dict[str, Any]], table_pk_map: Optional[Dict[str, List[str]]] = None):
        """
        Args:
            config: JSON 설정 딕셔너리 {컬럼명: {설정옵션들}}
        """
        self.cfg = config
        self.stats = {}  # 컬럼별 통계 저장: mean, std, clip_bounds 등
        self.is_fitted = False

        self.table_pk_map = table_pk_map


    def fit(self, df_train: pd.DataFrame) -> 'NumericPreprocessor':
        """
        훈련 데이터로부터 통계 계산 (기준 잡기)

        Args:
            df_train: 훈련 데이터프레임

        Returns:
            self
        """
        print("🔄 수치형 전처리기 학습 중...")

        for col in df_train.columns:
            if col not in self.cfg:
                continue

            col_cfg = self.cfg[col]
            col_data = df_train[col].copy()

            # 컬럼별 통계 초기화
            self.stats[col] = {}

            # 1. 결측값 처리를 위한 통계
            if col_cfg.get('fill') == 'median':
                self.stats[col]['fill_value'] = col_data.median()
            elif col_cfg.get('fill') == 'mode':
                self.stats[col]['fill_value'] = col_data.mode().iloc[0] if not col_data.mode().empty else 0
            elif isinstance(col_cfg.get('fill'), (int, float)):
                self.stats[col]['fill_value'] = col_cfg['fill']
            else:
                self.stats[col]['fill_value'] = 0

            # 결측값 채우기 (통계 계산을 위해)
            col_data = col_data.fillna(self.stats[col]['fill_value'])

            # 2. 클리핑 경계 계산
            if 'clip' in col_cfg:  # 백분위수 기반
                p_low, p_high = col_cfg['clip']
                self.stats[col]['clip_low'] = np.percentile(col_data, p_low)
                self.stats[col]['clip_high'] = np.percentile(col_data, p_high)
            elif 'clip_abs' in col_cfg:  # 절대값 기반
                self.stats[col]['clip_low'] = col_cfg['clip_abs'][0]
                self.stats[col]['clip_high'] = col_cfg['clip_abs'][1]

            # 클리핑 적용 (log1p, scale 계산을 위해)
            if 'clip_low' in self.stats[col]:
                if col_cfg.get('clip_to_null', False):
                    # 범위 밖은 null로 처리
                    mask = (col_data < self.stats[col]['clip_low']) | (col_data > self.stats[col]['clip_high'])
                    col_data[mask] = np.nan
                else:
                    # 범위 밖은 경계값으로 클리핑
                    col_data = col_data.clip(self.stats[col]['clip_low'], self.stats[col]['clip_high'])

            # 3. log1p 변환 (스케일링 통계 계산을 위해)
            if col_cfg.get('log1p', False):
                # log1p는 음수 처리를 위해 최소값 보정
                min_val = col_data.min()
                if min_val <= 0:
                    self.stats[col]['log1p_offset'] = abs(min_val) + 1
                    col_data_log = np.log1p(col_data + self.stats[col]['log1p_offset'])
                else:
                    self.stats[col]['log1p_offset'] = 0
                    col_data_log = np.log1p(col_data)
                col_data_for_scale = col_data_log
            else:
                col_data_for_scale = col_data

            # 4. 스케일링을 위한 통계
            if col_cfg.get('scale') == 'zscore':
                self.stats[col]['mean'] = col_data_for_scale.mean()
                self.stats[col]['std'] = col_data_for_scale.std()
                if self.stats[col]['std'] == 0:
                    self.stats[col]['std'] = 1  # 0으로 나누기 방지
            elif col_cfg.get('scale') == 'minmax':
                self.stats[col]['min'] = col_data_for_scale.min()
                self.stats[col]['max'] = col_data_for_scale.max()
                if self.stats[col]['min'] == self.stats[col]['max']:
                    self.stats[col]['max'] = self.stats[col]['min'] + 1  # 0으로 나누기 방지

        self.is_fitted = True
        print(f"✅ 수치형 전처리기 학습 완료: {len(self.stats)}개 컬럼")
        return self

    def transform(self, df: pd.DataFrame, table_name:str) -> pd.DataFrame:
        """
        학습된 통계로 데이터 변환

        Args:
            df: 변환할 데이터프레임
            table_name: 테이블 이름
        Returns:
            변환된 데이터프레임
        """
        if not self.is_fitted:
            raise ValueError("먼저 fit()을 호출해야 합니다.")

        original_order = list(df.columns)

        # 0) PK 확인 및 보존
        pk_cols = self.table_pk_map.get(table_name, [])
        if pk_cols:
            missing = [c for c in pk_cols if c not in df.columns]
            if missing:
                raise KeyError(f"[{table_name}] PK 컬럼 누락: {missing}")
        else:
            pk_cols = []

        processed = {}

        for pk in pk_cols:
            processed[pk] = df[pk].copy()

        for col in df.columns:
            if col in pk_cols:
                continue
            if col not in self.cfg or col not in self.stats:
                continue

            col_cfg = self.cfg[col]
            col_stats = self.stats[col]
            col_data = df[col].copy()

            # 1. 결측값 플래그 추가
            if col_cfg.get('add_flag', False):
                processed[f'{col}_is_null'] = col_data.isnull().astype('float32')

            # 2. 결측값 채우기
            col_data = col_data.fillna(col_stats['fill_value'])

            # 3. 클리핑
            if 'clip_low' in col_stats:
                if col_cfg.get('clip_to_null', False):
                    # 범위 밖은 null로 처리
                    mask = (col_data < col_stats['clip_low']) | (col_data > col_stats['clip_high'])
                    col_data[mask] = np.nan
                    # null로 처리된 값들을 fill_value로 재채우기
                    col_data = col_data.fillna(col_stats['fill_value'])
                else:
                    # 범위 밖은 경계값으로 클리핑
                    col_data = col_data.clip(col_stats['clip_low'], col_stats['clip_high'])

            # 4. log1p 변환
            if col_cfg.get('log1p', False):
                col_data = np.log1p(col_data + col_stats.get('log1p_offset', 0))

            # 5. 스케일링
            if col_cfg.get('scale') == 'zscore':
                col_data = (col_data - col_stats['mean']) / col_stats['std']
            elif col_cfg.get('scale') == 'minmax':
                col_data = (col_data - col_stats['min']) / (col_stats['max'] - col_stats['min'])

            # 6. float32로 변환
            processed[col] = col_data.astype('float32')

        # 원래 순서대로 DataFrame 구성
        df_out = pd.DataFrame({col: processed[col] for col in original_order if col in processed})

        # 추가적으로 만들어진 *_is_null 플래그 컬럼 뒤에 붙이기
        extra_cols = [c for c in processed.keys() if c not in df_out.columns]
        for c in extra_cols:
            df_out[c] = processed[c]

        return df_out


    def save(self, path: str):
        """전처리기 저장"""
        save_data = {
            'config': self.cfg,
            'stats': self.stats,
            'is_fitted': self.is_fitted
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"✅ 수치형 전처리기 저장 완료: {path}")

    @classmethod
    def load(cls, path: str) -> 'NumericPreprocessor':
        """전처리기 로드"""
        with open(path, 'rb') as f:
            save_data = pickle.load(f)

        preprocessor = cls(save_data['config'])
        preprocessor.stats = save_data['stats']
        preprocessor.is_fitted = save_data['is_fitted']

        print(f"✅ 수치형 전처리기 로드 완료: {path}")
        return preprocessor


def load_config(config_path: str) -> Dict[str, Dict[str, Any]]:
    """JSON 설정 파일 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def preprocess_numeric_data(df: pd.DataFrame, table_type: str) -> pd.DataFrame:
    """
    pipeline.py에서 호출되는 함수

    Args:
        df: 수치형 컬럼만 포함된 데이터프레임
        table_type: 'notice' 또는 'company'

    Returns:
        전처리된 데이터프레임
    """
    # 설정 파일 경로
    config_path = f"meta/{table_type}_numeric_config.json"


    try:
        # 설정 로드
        cfg = load_config(config_path)

        pk_cols = [c for c in TABLE_PK_MAP[table_type] if c in df.columns]

        # 실제 존재하는 컬럼만 필터링
        feat_cols = [c for c in cfg.keys() if c in df.columns]

        # ③ PK + 피처만 추출해 전처리기로 넘김
        use_cols = pk_cols + feat_cols
        work = df[use_cols].copy()

        # 전처리 수행 (학습 모드 - fit_transform 사용)
        preprocessor = NumericPreprocessor(cfg, TABLE_PK_MAP)
        processed_df = preprocessor.fit(work).transform(work, table_type)

        # 전처리기 저장 (나중에 동일한 변환을 위해)
        preprocessor.save(f"models/{table_type}_numeric_preprocessor.pkl")

        print(f"✅ {table_type} 수치형 전처리 완료: {len(cfg)}개 컬럼")
        return processed_df

    except Exception as e:
        print(f"❌ {table_type} 수치형 전처리 실패: {e}")
        return df


# 1) PK 보존을 고려한 서브셋 추출
def _collect_pk_and_numeric(df: pd.DataFrame, table_name: str, config: dict):
    # (a) PK 컬럼 확보 (인덱스에 있으면 reset_index)
    pk_cols = TABLE_PK_MAP.get(table_name, [])
    work = df

    # 인덱스에 PK가 숨어 있을 수 있으므로 복구
    if isinstance(work.index, pd.MultiIndex):
        if set(pk_cols).issubset(set([n for n in work.index.names if n])):
            work = work.reset_index()
    elif work.index.name in pk_cols:
        work = work.reset_index()

    # (b) 실제 존재하는 PK만 유지
    pk_cols = [c for c in pk_cols if c in work.columns]

    # (c) 수치형 대상 컬럼 (설정 파일 기준)
    numeric_cols = [col for col in config.keys() if col in work.columns]

    # (d) PK + 수치형 컬럼 합치기(중복 제거, PK가 앞에 오도록)
    use_cols = pk_cols + [c for c in numeric_cols if c not in pk_cols]

    # 안전장치: 최소 하나는 있어야 함
    if not use_cols:
        raise KeyError(f"[{table_name}] 사용할 컬럼이 없습니다. (PK:{TABLE_PK_MAP.get(table_name, [])}, cfg keys:{list(config.keys())[:5]}...)")

    return work[use_cols].copy(), pk_cols


if __name__ == "__main__":
    import os

    # CSV 파일로 테스트
    try:
        table_name = "notice"
        json_config_path = "meta/notice_numeric_config.json"

        # CSV 로드
        df = pd.read_csv("output/multiple/multiple_notices.csv")
        print(f"📄 CSV 로드 완료: {len(df)}행 × {len(df.columns)}열")

        # 수치형 컬럼만 추출 (설정 파일 기준)
        config = load_config(json_config_path)
        numeric_df, pk_cols = _collect_pk_and_numeric(df, table_name, config)

        # 전처리 실행
        print("\n" + "=" * 100)
        print("🔄 수치형 전처리 실행 중...")
        print("=" * 100)

        result = preprocess_numeric_data(numeric_df, table_name)

        print("\n" + "=" * 100)
        print("✅ 전처리 후 데이터")
        print("=" * 100)
        print(f"컬럼 수: {len(result.columns)}개")


        # 변화 요약
        print("\n" + "=" * 100)
        print("📋 전처리 변화 요약")
        print("=" * 100)

        original_cols = set(numeric_df.columns)
        new_cols = set(result.columns)
        added_cols = new_cols - original_cols

        print(f"원본 컬럼 수: {len(original_cols)}개")
        print(f"전처리 후 컬럼 수: {len(new_cols)}개")
        print(f"추가된 컬럼 수: {len(added_cols)}개")


        # 테스트용 저장
        os.makedirs("output/preprocessed", exist_ok=True)
        test_output_path = "output/preprocessed/notice_numeric_test.csv"
        result.to_csv(test_output_path, index=False, encoding='utf-8-sig')
        print(f"\n📁 테스트 결과 저장: {test_output_path}")

    except FileNotFoundError as e:
        print(f"❌ 파일을 찾을 수 없습니다: {e}")
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback

        traceback.print_exc()