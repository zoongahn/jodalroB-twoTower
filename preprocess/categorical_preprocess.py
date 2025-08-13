# categorical_preprocess.py
import pandas as pd
import numpy as np
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
import pickle

TABLE_PK_MAP = {
    'notice': ['bidntceno', 'bidntceord'],
    'company': ['bizno']
}

class CategoricalPreprocessor:
    """범주형 데이터 전처리 클래스"""

    def __init__(self, config: Dict[str, Dict[str, Any]], table_pk_map: Optional[Dict[str, List[str]]] = None):
        self.cfg = config
        self.mappings = {}  # 인코딩 매핑 정보 저장
        self.is_fitted = False
        self.table_pk_map = table_pk_map if table_pk_map else TABLE_PK_MAP

    def fit(self, df_train: pd.DataFrame) -> 'CategoricalPreprocessor':
        print("🔄 범주형 전처리기 학습 중...")
        for col in df_train.columns:
            if col not in self.cfg:
                continue

            col_cfg = self.cfg[col]
            col_data = df_train[col].copy()
            
            # 1. 결측값 처리
            null_strategy = col_cfg.get('null_strategy', 'new_category')
            if null_strategy == 'new_category':
                col_data.fillna('[NULL]', inplace=True)
            elif null_strategy == 'mode':
                mode_val = col_data.mode().iloc[0] if not col_data.mode().empty else '[MODE]'
                col_data.fillna(mode_val, inplace=True)
                self.mappings[f"{col}_mode"] = mode_val

            # 2. 희소 카테고리 처리
            rare_threshold = col_cfg.get('rare_threshold', 0.0)
            if rare_threshold > 0:
                counts = col_data.value_counts(normalize=True)
                rare_labels = counts[counts < rare_threshold].index
                col_data[col_data.isin(rare_labels)] = '[RARE]'
                self.mappings[f"{col}_rare_labels"] = list(rare_labels)

            # 3. 인코딩 매핑 생성
            encoding_method = col_cfg.get('encoding_method', 'label')
            if encoding_method == 'label':
                unique_labels = col_data.unique()
                self.mappings[col] = {label: i for i, label in enumerate(unique_labels)}
            elif encoding_method == 'frequency':
                 self.mappings[col] = col_data.value_counts(normalize=True).to_dict()

        self.is_fitted = True
        print(f"✅ 범주형 전처리기 학습 완료: {len(self.mappings)}개 매핑 생성")
        return self

    def transform(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("먼저 fit()을 호출해야 합니다.")

        df_out = pd.DataFrame(index=df.index)
        
        pk_cols = self.table_pk_map.get(table_name, [])
        for pk in pk_cols:
            if pk in df.columns:
                df_out[pk] = df[pk]

        for col in df.columns:
            if col not in self.cfg or col in pk_cols:
                continue

            col_cfg = self.cfg[col]
            col_data = df[col].copy()

            # 1. 결측값 플래그
            if col_cfg.get('add_flag', True):
                df_out[f'{col}_is_null'] = col_data.isnull().astype('float32')
            
            # 2. 결측값 처리
            null_strategy = col_cfg.get('null_strategy', 'new_category')
            if null_strategy == 'new_category':
                col_data.fillna('[NULL]', inplace=True)
            elif null_strategy == 'mode':
                 col_data.fillna(self.mappings.get(f"{col}_mode", '[MODE]'), inplace=True)

            # 3. 희소 카테고리 처리
            rare_labels = self.mappings.get(f"{col}_rare_labels")
            if rare_labels:
                col_data[col_data.isin(rare_labels)] = '[RARE]'

            # 4. 인코딩 적용
            encoding_method = col_cfg.get('encoding_method', 'label')
            mapping = self.mappings.get(col, {})
            
            unknown_strategy = col_cfg.get('unknown_strategy', 'new_category')
            
            if encoding_method == 'label':
                if unknown_strategy == 'new_category':
                    unknown_val = len(mapping) # 새로운 정수 값
                    df_out[col] = col_data.map(mapping).fillna(unknown_val).astype(int)
                else: # mode
                    mode_label = next(iter(mapping.keys())) # 가장 첫번째 값으로 대체
                    df_out[col] = col_data.map(mapping).fillna(mapping.get(mode_label, 0)).astype(int)
            
            elif encoding_method == 'frequency':
                df_out[col] = col_data.map(mapping).fillna(0).astype('float32')

        return df_out

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'config': self.cfg, 'mappings': self.mappings, 'is_fitted': self.is_fitted}, f)
        print(f"✅ 범주형 전처리기 저장 완료: {path}")

    @classmethod
    def load(cls, path: str) -> 'CategoricalPreprocessor':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        preprocessor = cls(data['config'])
        preprocessor.mappings = data['mappings']
        preprocessor.is_fitted = data['is_fitted']
        print(f"✅ 범주형 전처리기 로드 완료: {path}")
        return preprocessor

def load_config(config_path: str) -> Dict[str, Dict[str, Any]]:
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)
        
def preprocess_categorical_data(df: pd.DataFrame, table_type: str) -> pd.DataFrame:
    config_path = f"meta/{table_type}_categorical_config.json"
    try:
        cfg = load_config(config_path)
        
        preprocessor = CategoricalPreprocessor(cfg)
        processed_df = preprocessor.fit(df).transform(df, table_type)
        
        preprocessor.save(f"models/{table_type}_categorical_preprocessor.pkl")
        
        print(f"✅ {table_type} 범주형 전처리 완료.")
        return processed_df
    except Exception as e:
        print(f"❌ {table_type} 범주형 전처리 실패: {e}")
        return df

if __name__ == '__main__':
    try:
        table_name = "notice"
        json_config_path = "meta/notice_categorical_config.json"
        
        # 샘플 설정 파일 생성
        if not Path(json_config_path).exists():
            sample_cfg = {
                "prchsnm": {
                    "encoding_method": "label",
                    "add_flag": True,
                    "rare_threshold": 0.01,
                    "unknown_strategy": "new_category",
                    "null_strategy": "new_category"
                },
                "cntrctmthnm": {
                    "encoding_method": "frequency",
                     "null_strategy": "mode"
                }
            }
            with open(json_config_path, 'w', encoding='utf-8') as f:
                json.dump(sample_cfg, f, indent=2)
            print(f"📝 샘플 설정 파일 생성: {json_config_path}")

        df = pd.read_csv("output/multiple/multiple_notices.csv")
        
        config = load_config(json_config_path)
        cat_cols = [c for c in config.keys() if c in df.columns]
        
        result = preprocess_categorical_data(df[cat_cols], table_name)
        
        print("\\n✅ 전처리 후 데이터:")
        print(result.head())
        
        output_path = "output/preprocessed/notice_categorical_test.csv"
        result.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\\n📁 테스트 결과 저장: {output_path}")

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
