import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from sqlalchemy.engine import Engine
from tqdm import tqdm

from src.torchrec_preprocess.schema import TorchRecSchema
from src.torchrec_preprocess.torchrec_inputs import slice_and_convert_for_tower
from src.torchrec_preprocess.feature_store import build_feature_store
from src.torchrec_preprocess.feature_projector import FeatureProjector


class BidDataset(Dataset):
    """
    입찰 데이터 기반 Positive Pair Dataset
    """
    def __init__(
        self, 
        db_engine: Engine, 
        schema: TorchRecSchema, 
        limit: Optional[int] = None,
        test_split: float = 0.1,
        is_train: bool = True,
        shuffle_seed: int = 42,
        load_all_features: bool = True
    ):
        """
        Args:
            db_engine: 데이터베이스 연결
            schema: TwoTowerSchema 객체
            limit: 로딩할 pair 수 제한 (None이면 전체)
            test_split: 테스트 데이터 비율
            is_train: True면 훈련용, False면 테스트용
            shuffle_seed: 데이터 섞기 위한 시드
        """
        self.db_engine = db_engine
        self.schema = schema
        self.is_train = is_train
        self.load_all_features = load_all_features
        
        print(f"BidDataset 초기화 중... (train={is_train})")
        
        # 1) Positive pair 로딩
        self.pairs = self._load_positive_pairs(limit, test_split, shuffle_seed)
        
        # 2) Notice와 Company 피처 데이터 로딩
        self._load_features()
        
        print(f"총 {len(self.pairs)}개 positive pair 로딩 완료")
        
    def _load_positive_pairs(self, limit: Optional[int], test_split: float, shuffle_seed: int) -> pd.DataFrame:
        """bid_two_tower 테이블에서 positive pair 로딩"""
        query = f"""
        SELECT bidntceno, bidntceord, bizno 
        FROM {self.schema.pair.table}
        """
        
        if limit:
            query += f" LIMIT {limit}"
            
        print(f"Positive pair 로딩: {query}")
        pairs_df = pd.read_sql(query, self.db_engine)
        
        # 데이터 섞기
        pairs_df = pairs_df.sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)

        # test_split이 0이면 전체 데이터를 train으로 사용
        if test_split == 0:
            print(f"전체 데이터를 훈련용으로 사용: {len(pairs_df)}개")
            return pairs_df
            
        # Train/Test 분할
        n_total = len(pairs_df)
        n_test = int(n_total * test_split)
        
        if self.is_train:
            pairs_df = pairs_df[n_test:].reset_index(drop=True)
            print(f"훈련용 데이터: {len(pairs_df)}개")
        else:
            pairs_df = pairs_df[:n_test].reset_index(drop=True)
            print(f"테스트용 데이터: {len(pairs_df)}개")
            
        return pairs_df
    
    def _load_features(self):
        """Notice와 Company 피처 데이터 로딩 (전체 또는 선택적)"""
        if self.load_all_features:
            print("전체 피처 데이터 로딩 중...")
            self._load_all_features()
        else:
            print("선택적 피처 데이터 로딩 중...")
            self._load_selective_features()
    
    def _load_selective_features(self):
        """Positive pair에 포함된 Notice와 Company만 선택적으로 로딩"""
        # 1) Positive pair에서 unique한 ID들 추출
        unique_notices = self.pairs[['bidntceno', 'bidntceord']].drop_duplicates()
        unique_companies = self.pairs[['bizno']].drop_duplicates()
        
        print(f"[DEBUG] 필요한 Notice: {len(unique_notices)}개")
        print(f"[DEBUG] 필요한 Company: {len(unique_companies)}개")
        
        # 2) Notice 피처 선택적 로딩
        print("Notice 피처 선택적 로딩...")
        notice_store = self._build_selective_feature_store(
            schema=self.schema.notice,
            id_df=unique_notices,
            id_columns=['bidntceno', 'bidntceord']
        )
        
        # 3) Company 피처 선택적 로딩  
        print("Company 피처 선택적 로딩...")
        company_store = self._build_selective_feature_store(
            schema=self.schema.company,
            id_df=unique_companies,
            id_columns=['bizno']
        )
        
        # 4) ID 매핑 및 projector 설정
        self._build_id_mappings(notice_store, company_store)
        self.notice_store = notice_store
        self.company_store = company_store
        self._setup_projectors()
        
        print("선택적 피처 데이터 로딩 완료")
    
    def _build_selective_feature_store(self, schema, id_df, id_columns):
        """특정 ID들에 해당하는 피처만 로딩"""
        print(f"[DEBUG] 요청하는 ID 개수: {len(id_df)}")

        # WHERE 조건 생성
        if len(id_columns) == 1:
            id_list = id_df[id_columns[0]].astype(str).tolist()
            where_clause = f"{id_columns[0]} IN ({','.join(repr(x) for x in id_list)})"
        else:
            conditions = []
            for _, row in id_df.iterrows():
                condition_parts = []
                for col in id_columns:
                    condition_parts.append(f"{col}='{row[col]}'")
                conditions.append(f"({' AND '.join(condition_parts)})")
            where_clause = f"{' OR '.join(conditions)}"
        
        from src.torchrec_preprocess.feature_store import build_feature_store_with_condition
        
        result = build_feature_store_with_condition(
            self.db_engine, 
            schema, 
            where_clause, 
            chunksize=5000, 
            show_progress=True
        )
        
        print(f"[DEBUG] 반환된 결과: {type(result)}")
        if result and 'ids' in result:
            ids = result['ids']
            print(f"[DEBUG] 반환된 ID 개수: {len(ids)}")
            if len(id_columns) == 1:  # company의 경우
                id_values = [str(id_val) if not isinstance(id_val, (list, tuple)) else str(id_val[0]) for id_val in ids]
                
                # 요청 ID와 반환 ID 집합 비교
                requested_set = set(id_list)
                returned_set = set(id_values)
                
                missing_ids = requested_set - returned_set
                extra_ids = returned_set - requested_set
                
                print(f"[DEBUG] === ID 집합 비교 ===")
                print(f"[DEBUG] 누락된 ID 수: {len(missing_ids)} | 추가된 ID 수: {len(extra_ids)}")
                
                if missing_ids:
                    print(f"[DEBUG] 누락된 ID들: {list(missing_ids)[:10]}...")  # 처음 10개만
                if extra_ids:
                    print(f"[DEBUG] 추가된 ID들: {list(extra_ids)[:10]}...")  # 처음 10개만
                    
            else:  # notice의 경우 (복합 키)
                # 복합 키는 튜플로 비교
                requested_tuples = set()
                for _, row in id_df.iterrows():
                    key_tuple = tuple(str(row[col]) for col in id_columns)
                    requested_tuples.add(key_tuple)
                
                returned_tuples = set()
                for id_tuple in ids:
                    if isinstance(id_tuple, (list, tuple)):
                        key_tuple = tuple(str(x) for x in id_tuple)
                    else:
                        key_tuple = (str(id_tuple),)
                    returned_tuples.add(key_tuple)
                
                missing_tuples = requested_tuples - returned_tuples
                extra_tuples = returned_tuples - requested_tuples
                
                print(f"[DEBUG] === ID 집합 비교 (복합 키) ===")
                print(f"[DEBUG] 누락된 키 수: {len(missing_tuples)} | 추가된 키 수: {len(extra_tuples)}")
                
                if missing_tuples:
                    print(f"[DEBUG] 누락된 키들: {list(missing_tuples)[:5]}...")
                if extra_tuples:
                    print(f"[DEBUG] 추가된 키들: {list(extra_tuples)[:5]}...")
                
        return result
    
    def _build_feature_store_pandas(self, schema, where_clause):
        """Pandas 기반 선택적 피처 로딩"""
        # 모든 컬럼 선택
        all_columns = schema.pk_cols + schema.numeric + schema.categorical + schema.text
        column_str = ", ".join(all_columns)
        
        query = f"SELECT {column_str} FROM {schema.table} {where_clause}"
        print(f"쿼리 실행: {query[:100]}...")
        
        df = pd.read_sql(query, self.db_engine)
        
        # build_feature_store와 유사한 구조로 반환
        result = {
            'ids': df[schema.pk_cols].values.tolist(),
            'numeric': df[schema.numeric].values if schema.numeric else None,
            'categorical': df[schema.categorical].values if schema.categorical else None,
            'text': {col: df[col].values for col in schema.text} if schema.text else {}
        }
        
    def _load_all_features(self):
        """Notice와 Company 전체 피처 데이터 미리 로딩 (기존 방식)"""        
        # Notice 피처 로딩
        print("Notice 피처 로딩...")
        notice_store = build_feature_store(
            self.db_engine, 
            self.schema.notice, 
            chunksize=5000, 
            limit=None, 
            show_progress=True
        )
        
        # Company 피처 로딩
        print("Company 피처 로딩...")
        company_store = build_feature_store(
            self.db_engine, 
            self.schema.company, 
            chunksize=5000, 
            limit=None, 
            show_progress=True
        )
        
        # ID를 키로 하는 인덱스 매핑 생성
        self._build_id_mappings(notice_store, company_store)
        
        # 피처 데이터 저장
        self.notice_store = notice_store
        self.company_store = company_store
        
        self._setup_projectors()
        print("전체 피처 데이터 로딩 완료")
        
    def _setup_projectors(self):
        """Projector 설정"""
        self.notice_projector = FeatureProjector(
            num_dim=len(self.schema.notice.numeric),
            text_dim=768,
            num_proj_dim=128,
            text_proj_dim=128
        )
        self.company_projector = FeatureProjector(
            num_dim=len(self.schema.company.numeric),
            text_dim=768,
            num_proj_dim=128,
            text_proj_dim=128
        )
            
    def _build_id_mappings(self, notice_store: Dict, company_store: Dict):
            """ID를 데이터 인덱스로 매핑하는 딕셔너리 생성"""
            print("ID 매핑 테이블 생성 중...")
            
            # Notice ID 매핑 (bidntceno, bidntceord 조합)
            notice_ids = notice_store['ids']  # [(bidntceno, bidntceord), ...]
            print(f"[DEBUG] Notice IDs 타입: {type(notice_ids)}")
            self.notice_id_to_idx = {tuple(id_pair): idx for idx, id_pair in enumerate(notice_ids)}
            
            # Company ID 매핑 (bizno) - 디버깅 추가
            company_ids = company_store['ids']  # [bizno, ...]
            print(f"[DEBUG] Company IDs 타입: {type(company_ids)}")
            print(f"[DEBUG] Company IDs 길이: {len(company_ids)}")
            
            if company_ids:
                print(f"[DEBUG] 첫 번째 Company ID 타입: {type(company_ids[0])}")
                # Company는 single primary key이므로 튜플에서 첫 번째 원소 추출
                if isinstance(company_ids[0], (tuple, list)):
                    # 튜플/리스트인 경우 첫 번째 원소만 사용
                    self.company_id_to_idx = {str(id_tuple[0]): idx for idx, id_tuple in enumerate(company_ids)}
                else:
                    # 이미 단일 값인 경우
                    self.company_id_to_idx = {str(company_id): idx for idx, company_id in enumerate(company_ids)}
                    print(f"[DEBUG] 단일값 Company ID 샘플: {[str(company_id) for company_id in company_ids[:3]]}")
            
            print(f"Notice ID 매핑: {len(self.notice_id_to_idx)}개")
            print(f"Company ID 매핑: {len(self.company_id_to_idx)}개")
            
            # 매핑 딕셔너리의 키 샘플 출력
            if self.company_id_to_idx:
                sample_keys = list(self.company_id_to_idx.keys())[:3]
                print(f"[DEBUG] Company 매핑 딕셔너리 키 샘플: {sample_keys} ...")
                print(f"[DEBUG] Company 매핑 딕셔너리 키 타입: {[type(k) for k in sample_keys]} ...")
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:        
        """배치 아이템 반환"""
        pair = self.pairs.iloc[idx]
        
        # Pair 정보 추출
        bidntceno = pair['bidntceno']
        bidntceord = pair['bidntceord']
        bizno = pair['bizno']
        
        # Notice 인덱스 찾기
        notice_key = (bidntceno, bidntceord)
        if notice_key not in self.notice_id_to_idx:
            raise KeyError(f"Notice ID {notice_key}를 찾을 수 없습니다")
        notice_idx = self.notice_id_to_idx[notice_key]

            
        if bizno not in self.company_id_to_idx:
            print(f"[ERROR] 찾을 수 없는 bizno: {bizno}")
            print(f"[ERROR] 현재 pairs의 unique company ID 수: {len(self.pairs['bizno'].unique())}")
            print(f"[ERROR] 매핑 딕셔너리 ID 수: {len(self.company_id_to_idx)}")
            
            # 현재 pairs의 unique company ID들과 매핑 딕셔너리 비교
            pairs_company_ids = set(str(x) for x in self.pairs['bizno'].unique())
            mapping_company_ids = set(self.company_id_to_idx.keys())
            
            missing_in_mapping = pairs_company_ids - mapping_company_ids
            extra_in_mapping = mapping_company_ids - pairs_company_ids
            
            print(f"[ERROR] pairs에는 있지만 매핑에 없는 ID 수: {len(missing_in_mapping)}")
            print(f"[ERROR] 매핑에는 있지만 pairs에 없는 ID 수: {len(extra_in_mapping)}")
            
            if missing_in_mapping:
                print(f"[ERROR] 누락된 ID 예시: {list(missing_in_mapping)[:10]}")
                
            raise KeyError(f"Company ID {bizno}를 찾을 수 없습니다")
        
        company_idx = self.company_id_to_idx[bizno]
        
        # Notice 피처 추출
        notice_input = slice_and_convert_for_tower(
            store_result=self.notice_store,
            row_idx=[notice_idx],  # 단일 샘플
            categorical_keys=self.schema.notice.categorical,
            text_cols=self.schema.notice.text,
            projector=self.notice_projector,
            fuse_projected=True
        )
        
        # Company 피처 추출
        company_input = slice_and_convert_for_tower(
            store_result=self.company_store,
            row_idx=[company_idx],  # 단일 샘플
            categorical_keys=self.schema.company.categorical,
            text_cols=self.schema.company.text,
            projector=self.company_projector,
            fuse_projected=True
        )
        
        return {
            "notice": {
                "dense": notice_input["dense"],  # [1, dense_dim]
                "kjt": notice_input["kjt"]
            },
            "company": {
                "dense": company_input["dense"],  # [1, dense_dim]
                "kjt": company_input["kjt"]
            },
            "pair_info": {
                "bidntceno": bidntceno,
                "bidntceord": bidntceord,
                "bizno": bizno
            }
        }


def bid_collate_fn(batch: List[Dict]) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    배치를 TwoTowerTrainTask에서 사용할 수 있는 형태로 변환
    """
    batch_size = len(batch)
    
    # Notice 데이터 수집
    notice_dense_list = []
    notice_kjt_list = []
    
    # Company 데이터 수집
    company_dense_list = []
    company_kjt_list = []
    
    for item in batch:
        notice_dense_list.append(item["notice"]["dense"])
        notice_kjt_list.append(item["notice"]["kjt"])
        company_dense_list.append(item["company"]["dense"])
        company_kjt_list.append(item["company"]["kjt"])
    
    # Dense 텐서 배치로 결합
    notice_dense_batch = torch.cat(notice_dense_list, dim=0)  # [B, dense_dim]
    company_dense_batch = torch.cat(company_dense_list, dim=0)  # [B, dense_dim]
    
    # KJT 배치로 결합 (간단한 구현 - 실제로는 더 복잡할 수 있음)
    from torchrec import KeyedJaggedTensor
    
    # Notice KJT 결합
    notice_values_list = []
    notice_lengths_list = []
    for kjt in notice_kjt_list:
        notice_values_list.append(kjt.values())
        notice_lengths_list.append(kjt.lengths())
    
    notice_values_batch = torch.cat(notice_values_list)
    notice_lengths_batch = torch.cat(notice_lengths_list)
    
    notice_kjt_batch = KeyedJaggedTensor.from_lengths_sync(
        keys=notice_kjt_list[0].keys(),  # 첫 번째 아이템의 키 사용
        values=notice_values_batch,
        lengths=notice_lengths_batch
    )
    
    # Company KJT 결합 (동일한 방식)
    company_values_list = []
    company_lengths_list = []
    for kjt in company_kjt_list:
        company_values_list.append(kjt.values())
        company_lengths_list.append(kjt.lengths())
    
    company_values_batch = torch.cat(company_values_list)
    company_lengths_batch = torch.cat(company_lengths_list)
    
    company_kjt_batch = KeyedJaggedTensor.from_lengths_sync(
        keys=company_kjt_list[0].keys(),
        values=company_values_batch,
        lengths=company_lengths_batch
    )
    
    return {
        "notice": {
            "dense": notice_dense_batch,
            "kjt": notice_kjt_batch
        },
        "company": {
            "dense": company_dense_batch,
            "kjt": company_kjt_batch
        }
    }


def create_bid_dataloaders(
    db_engine: Engine,
    schema: TorchRecSchema,
    batch_size: int = 32,
    limit: Optional[int] = None,
    test_split: float = 0.1,
    shuffle_seed: int = 42,
    num_workers: int = 0,
    load_all_features: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    훈련용/테스트용 DataLoader 생성
    
    Args:
        load_all_features: True면 전체 피처 로딩(느림), False면 선택적 로딩(빠름)
    
    Returns:
        (train_dataloader, test_dataloader)
    """
    # Dataset 생성
    train_dataset = BidDataset(
        db_engine=db_engine,
        schema=schema,
        limit=limit,
        test_split=test_split,
        is_train=True,
        shuffle_seed=shuffle_seed,
        load_all_features=load_all_features
    )
    
    # Test Dataset은 test_split > 0일 때만 생성
    test_dataset = None
    if test_split > 0:
        test_dataset = BidDataset(
            db_engine=db_engine,
            schema=schema,
            limit=limit,
            test_split=test_split,
            is_train=False,
            shuffle_seed=shuffle_seed,
            load_all_features=load_all_features
        )
    
    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=bid_collate_fn,
        num_workers=num_workers,
        pin_memory=False
    )
    
    # Test DataLoader 생성 (test_dataset이 있을 때만)
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=bid_collate_fn,
            num_workers=num_workers,
            pin_memory=False
        )
    
    return train_loader, test_loader


# 사용 예시
def test_bid_dataloader() -> Tuple[DataLoader, Optional[DataLoader]]:
    """BidDataLoader 테스트"""
    from data.database_connector import DatabaseConnector
    from src.torchrec_preprocess.schema import build_torchrec_schema_from_meta
    
    # DB 연결
    db = DatabaseConnector()
    engine = db.engine
    
    # 스키마 구축
    config = {
        "notice_table": "notice",
        "company_table": "company", 
        "pair_table": "bid_two_tower",
        "pair_notice_id_cols": ["bidntceno", "bidntceord"],
        "pair_company_id_cols": ["bizno"],
        "metadata_path": "meta/metadata.csv"
    }
    schema = build_torchrec_schema_from_meta(**config)
    
    print("=== 훈련 전용 모드 (테스트셋 없음) ===")
    train_loader, test_loader = create_bid_dataloaders(
        db_engine=engine,
        schema=schema,
        batch_size=15,
        limit=10000,
        test_split=0.1,  # 테스트셋 생성하지 않음
        load_all_features=False
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader) if test_loader else 'None'}")
    
    # 첫 번째 배치 테스트
    for batch in train_loader:
        print("Batch shapes:")
        print(f"  NOTICE dense: {batch['notice']['dense'].shape}")
        print(f"  NOTICE KJT keys: {len(batch['notice']['kjt'].keys())}")
        print(f"  COMPANY dense: {batch['company']['dense'].shape}")
        print(f"  COMPANY KJT keys: {len(batch['company']['kjt'].keys())}")
        break
    
    return train_loader, test_loader


if __name__ == "__main__":
    test_bid_dataloader()