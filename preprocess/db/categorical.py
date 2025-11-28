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

# 특수 토큰 정의
NULL_TOKEN = '[NULL]'
RARE_TOKEN = '[RARE]'
UNKNOWN_TOKEN = '[UNKNOWN]'


class CategoricalPreprocessor:
    """
    범주형 데이터 전처리 클래스 (Embedding Layer 최적화)
    - 모든 변수를 정수 인덱스로 변환합니다.
    - 희소/결측/미지 카테고리에 대한 안정적인 처리를 보장합니다.
    - 모델링에 필요한 모든 메타데이터를 json으로 저장합니다.
    """

    def __init__(self, config: Dict[str, Dict[str, Any]], table_pk_map: Optional[Dict[str, List[str]]] = None):
        self.cfg = config
        self.mappings = {}  # 인코딩 매핑 및 메타데이터 저장
        self.is_fitted = False
        self.table_pk_map = table_pk_map if table_pk_map else TABLE_PK_MAP

    def fit(self, df_train: pd.DataFrame) -> 'CategoricalPreprocessor':
        print("🔄 전문가 모드: 범주형 전처리기 학습 시작...")
        for col in df_train.columns:
            if col not in self.cfg:
                continue

            col_cfg = self.cfg[col]
            col_data = df_train[col].copy().astype(str) # 안정성을 위해 문자열로 변환

            # 1. 결측값 처리
            has_null = col_data.isnull().any()
            if has_null:
                col_data.fillna(NULL_TOKEN, inplace=True)

            # 2. 희소 카테고리 처리 (Rare Label Handling)
            # 카디널리티가 너무 높은 변수의 노이즈를 줄여 모델 안정성 및 성능 향상
            rare_threshold = col_cfg.get('rare_threshold', 0.0)
            if rare_threshold > 0:
                counts = col_data.value_counts(normalize=True)
                rare_labels = counts[counts < rare_threshold].index
                if len(rare_labels) > 0:
                    col_data[col_data.isin(rare_labels)] = RARE_TOKEN

            # 3. 최종 어휘(Vocabulary) 생성 및 인덱싱
            # UNKNOWN 토큰을 항상 어휘에 포함하여, 추론 시 새로운 카테고리에 대응
            unique_labels = col_data.unique().tolist()
            vocab = sorted([str(label) for label in unique_labels])
            
            # UNKNOWN 토큰이 어휘에 없으면 추가
            if UNKNOWN_TOKEN not in vocab:
                vocab.append(UNKNOWN_TOKEN)

            # 최종 매핑 정보 및 메타데이터 저장
            self.mappings[col] = {
                'map': {label: i for i, label in enumerate(vocab)},
                'vocab_size': len(vocab),
                'has_null': has_null,
                'has_rare': rare_threshold > 0 and len(rare_labels) > 0,
                'unknown_token_idx': vocab.index(UNKNOWN_TOKEN)
            }

        self.is_fitted = True
        print(f"✅ 학습 완료: {len(self.mappings)}개 변수에 대한 어휘 및 메타데이터 생성")
        return self

    def transform(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("먼저 fit()을 호출해야 합니다. 모델링과 전처리의 기준이 다르면 예측이 망가집니다.")

        df_out = pd.DataFrame(index=df.index)
        
        pk_cols = self.table_pk_map.get(table_name, [])
        for pk in pk_cols:
            if pk in df.columns:
                df_out[pk] = df[pk]

        for col in df.columns:
            if col not in self.cfg or col in pk_cols:
                continue

            col_cfg = self.cfg[col]
            col_data = df[col].copy().astype(str)
            
            # 1. 결측값 플래그 (모델에게 결측 정보를 명시적으로 알려주는 것은 중요한 피처가 될 수 있음)
            df_out[f'{col}_is_null'] = df[col].isnull().astype('float32')
            
            # 2. 결측값 토큰화
            col_data.fillna(NULL_TOKEN, inplace=True)
            
            # 3. 학습된 매핑으로 인코딩
            mapping_info = self.mappings.get(col, {})
            mapping = mapping_info.get('map', {})
            unknown_idx = mapping_info.get('unknown_token_idx', -1) # -1은 에러 확인용

            # map에 없는 새로운 값은 모두 unknown_idx로 대체
            df_out[col] = col_data.map(mapping).fillna(unknown_idx).astype(int)

        return df_out

    def save(self, path: str):
        """
        전처리기 객체(.pkl)와 모델링을 위한 메타데이터(.json)를 함께 저장
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # 1. Pickle 파일: 전처리기 전체 객체 저장
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"✅ 범주형 전처리기 저장 완료: {path}")

        # 2. JSON 파일: 모델러를 위한 명세서(_is_null 컬럼은 항상 생성되므로 명시 불필요)
        metadata_path = Path(path).with_suffix('.json')
        model_metadata = {
            col: {
                # 임베딩 레이어의 input_dim 으로 사용
                'input_dim': info.get('vocab_size', 0), 
                # UNKNOWN 토큰이 어휘에 포함되었는지 여부
                'has_unknown_token': UNKNOWN_TOKEN in info.get('map', {}),
                # NULL 토큰이 어휘에 포함되었는지 여부
                'has_null_token': NULL_TOKEN in info.get('map', {}),
                 # RARE 토큰이 어휘에 포함되었는지 여부
                'has_rare_token': RARE_TOKEN in info.get('map', {})
            } for col, info in self.mappings.items()
        }
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(model_metadata, f, indent=4)
        print(f"✅ 모델링 메타데이터 저장 완료: {metadata_path}")

    @classmethod
    def load(cls, path: str) -> 'CategoricalPreprocessor':
        with open(path, 'rb') as f:
            preprocessor = pickle.load(f)
        print(f"✅ 범주형 전처리기 로드 완료: {path}")
        return preprocessor

def load_config(config_path: str) -> Dict[str, Dict[str, Any]]:
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)
        
def preprocess_categorical_data(df: pd.DataFrame, table_type: str) -> pd.DataFrame:
    config_path = f"preprocess/db/config/{table_type}_categorical_config.json"
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
        json_config_path = "preprocess/db/config/notice_categorical_config.json"
        
        # 샘플 설정 파일 생성
        if not Path(json_config_path).exists():
            sample_cfg = {
                "bidmethdnm": {
                    "encoding_method": "label",
                    "add_flag": True,
                    "null_strategy": "new_category"
                },
                "cntrctcnclsmthdnm": {
                    "encoding_method": "label",
                     "null_strategy": "mode"
                }
            }
            with open(json_config_path, 'w', encoding='utf-8') as f:
                json.dump(sample_cfg, f, indent=2)
            print(f"📝 샘플 설정 파일 생성: {json_config_path}")

        df = pd.read_csv("output/multiple/multiple_notices.csv")
        
        config = load_config(json_config_path)
        
        # 설정 파일에 정의된 컬럼 중 실제 데이터에 있는 것만 선택
        cat_cols = [c for c in config.keys() if c in df.columns]
        
        # PK 컬럼 추가
        pk_cols = [pk for pk in TABLE_PK_MAP.get(table_name, []) if pk in df.columns]
        
        # 전처리에 사용할 컬럼 목록 (PK + 범주형)
        process_cols = pk_cols + cat_cols
        
        print(f"🔄 PK {pk_cols}와 범주형 {cat_cols} 컬럼 전처리를 시작합니다.")
        
        result = preprocess_categorical_data(df[process_cols], table_name)
        
        print("\\n✅ 전처리 후 데이터:")
        print(result.head())
        
        output_path = "output/preprocessed/notice_categorical_test.csv"
        result.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\\n📁 테스트 결과 저장: {output_path}")

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
