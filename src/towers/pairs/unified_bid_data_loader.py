import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from sqlalchemy.engine import Engine
from tqdm import tqdm

from src.torchrec_preprocess.schema import TorchRecSchema
from src.torchrec_preprocess.torchrec_inputs import slice_and_convert_for_tower
from src.torchrec_preprocess.feature_store import build_feature_store, build_feature_store_with_condition
from src.torchrec_preprocess.feature_projector import FeatureProjector


class UnifiedBidDataset(Dataset):
    """
    통합 입찰 데이터셋 - 스트리밍과 선택적 피처 로딩 지원
    
    4가지 모드 지원:
    1. streaming=False, load_all_features=True: 기존 방식
    2. streaming=False, load_all_features=False: selective loading
    3. streaming=True, load_all_features=True: pair만 스트리밍
    4. streaming=True, load_all_features=False: 완전 스트리밍 (권장)
    """
    def __init__(
        self, 
        db_engine: Engine, 
        schema: TorchRecSchema, 
        limit: Optional[int] = None,
        test_split: float = 0.1,
        is_train: bool = True,
        shuffle: bool = False,
        shuffle_seed: int = 42,
        streaming: bool = False,
        chunk_size: int = 1000,
        load_all_features: bool = True,
        feature_chunksize: int = 5000,  # 추가: 피처 로딩 시 청크 사이즈
        feature_limit: Optional[int] = None,  # 추가: 피처 로딩 시 제한
        shared_feature_stores: Optional[Dict] = None
    ):
        """
        Args:
            streaming: True면 pair를 청크 단위로 스트리밍
            chunk_size: 스트리밍 시 청크 크기
            load_all_features: False면 필요한 피처만 선택적 로딩
        """
        self.db_engine = db_engine
        self.schema = schema
        self.is_train = is_train
        self.streaming = streaming
        self.chunk_size = chunk_size
        self.load_all_features = load_all_features
        self.feature_chunksize = feature_chunksize
        self.feature_limit = feature_limit
        self.shared_feature_stores = shared_feature_stores
        
        print(f"UnifiedBidDataset 초기화... (train={is_train}, streaming={streaming}, load_all_features={load_all_features})")
        
        if streaming:
            # 스트리밍 모드: 개수와 범위만 계산
            self.total_count, self.start_id, self.end_id = self._get_data_info_streaming(limit, test_split)
            self.chunk_cache = {}  # {chunk_id: DataFrame}
            self.pairs = None  # 스트리밍에서는 pairs를 미리 로드하지 않음
        else:
            # 기존 모드: 전체 pairs 로드
            self.pairs = self._load_positive_pairs_static(limit, test_split, shuffle, shuffle_seed)
            self.total_count = len(self.pairs)
        
        # 피처 로딩
        if load_all_features:
            if shared_feature_stores is not None:
                print("공유 피처 스토어 사용")
                self.notice_store = shared_feature_stores['notice']
                self.company_store = shared_feature_stores['company']
                self._build_id_mappings(self.notice_store, self.company_store)
                self._setup_projectors()
            else:
                print("개별 피처 스토어 로딩")
                self._load_all_features()
        else:
            if streaming:
                print("스트리밍 모드에서는 피처를 청크별로 동적 로딩합니다.")
                self._prepare_dynamic_feature_loading()
            else:
                self._load_selective_features_static()
        
        print(f"총 {self.total_count}개 pair 준비 완료")
        
    def _get_data_info_streaming(self, limit: Optional[int], test_split: float) -> Tuple[int, int, int]:
        """스트리밍용 데이터 정보 조회"""
        if limit:
            count_query = f"""
            SELECT COUNT(*) as total_count, MIN(id) as min_id, MAX(id) as max_id
            FROM (
                SELECT id FROM {self.schema.pair.table} 
                ORDER BY id LIMIT {limit}
            ) subq
            """
        else:
            count_query = f"""
            SELECT COUNT(*) as total_count, MIN(id) as min_id, MAX(id) as max_id
            FROM {self.schema.pair.table}
            """
        
        result = pd.read_sql(count_query, self.db_engine).iloc[0]
        total_count = int(result['total_count'])
        min_id = int(result['min_id'])
        max_id = int(result['max_id'])
        
        if test_split == 0:
            return total_count, min_id, max_id
        
        n_test = int(total_count * test_split)
        n_train = total_count - n_test
        
        if self.is_train:
            start_offset = n_test
            id_query = f"""
            SELECT id FROM {self.schema.pair.table} 
            ORDER BY id OFFSET {start_offset} LIMIT 1
            """
            start_id = int(pd.read_sql(id_query, self.db_engine).iloc[0]['id'])
            return n_train, start_id, max_id
        else:
            return n_test, min_id, min_id + n_test - 1
    
    def _load_positive_pairs_static(self, limit: Optional[int], test_split: float, shuffle: bool, shuffle_seed: int) -> pd.DataFrame:
        """기존 방식의 static pair 로딩"""
        query = f"""
        SELECT bidntceno, bidntceord, bizno 
        FROM {self.schema.pair.table}
        ORDER BY id
        """
        
        if limit:
            query += f" LIMIT {limit}"
            
        pairs_df = pd.read_sql(query, self.db_engine)
        
        if shuffle:
            pairs_df = pairs_df.sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)
            print(f"데이터셋 내부 셔플 적용 (seed={shuffle_seed})")
        else:
            print("데이터셋 내부 셔플 미적용")
            
        
        if test_split == 0:
            return pairs_df
            
        n_total = len(pairs_df)
        n_test = int(n_total * test_split)
        
        if self.is_train:
            pairs_df = pairs_df[n_test:].reset_index(drop=True)
        else:
            pairs_df = pairs_df[:n_test].reset_index(drop=True)
            
        return pairs_df
    
    def _load_all_features(self):
        """모든 피처 미리 로딩"""
        print("전체 피처 데이터 로딩 중...")
        
        notice_store = build_feature_store(
            self.db_engine, self.schema.notice, 
            chunksize=self.feature_chunksize, limit=self.feature_limit, show_progress=True
        )
        
        company_store = build_feature_store(
            self.db_engine, self.schema.company, 
            chunksize=self.feature_chunksize, limit=self.feature_limit, show_progress=True
        )
        
        self._build_id_mappings(notice_store, company_store)
        self.notice_store = notice_store
        self.company_store = company_store
        self._setup_projectors()
        
    def _prepare_dynamic_feature_loading(self):
        """동적 피처 로딩 준비 (스트리밍 + selective)"""
        self.feature_cache = {}  # {chunk_id: {'notice_store': ..., 'company_store': ...}}
        self.notice_store = None
        self.company_store = None
        self._setup_projectors()
        
    def _load_selective_features_static(self):
        """정적 selective 피처 로딩 (non-streaming)"""
        unique_notices = self.pairs[['bidntceno', 'bidntceord']].drop_duplicates()
        unique_companies = self.pairs[['bizno']].drop_duplicates()
        
        notice_store = self._build_selective_feature_store(
            self.schema.notice, unique_notices, ['bidntceno', 'bidntceord']
        )
        company_store = self._build_selective_feature_store(
            self.schema.company, unique_companies, ['bizno']
        )
        
        self._build_id_mappings(notice_store, company_store)
        self.notice_store = notice_store
        self.company_store = company_store
        self._setup_projectors()
        
    def _load_selective_features_for_chunk(self, chunk_df: pd.DataFrame) -> Tuple[Dict, Dict]:
        """청크별 selective 피처 로딩"""
        unique_notices = chunk_df[['bidntceno', 'bidntceord']].drop_duplicates()
        unique_companies = chunk_df[['bizno']].drop_duplicates()
        
        notice_store = self._build_selective_feature_store(
            self.schema.notice, unique_notices, ['bidntceno', 'bidntceord']
        )
        company_store = self._build_selective_feature_store(
            self.schema.company, unique_companies, ['bizno']
        )
        
        return notice_store, company_store
        
    def _build_selective_feature_store(self, schema, id_df, id_columns):
        """선택적 피처 스토어 구축"""
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
        
        return build_feature_store_with_condition(
            self.db_engine, schema, where_clause, 
            chunksize=self.feature_chunksize, show_progress=False
        )
    
    def _setup_projectors(self):
        """Projector 설정"""
        self.notice_projector = FeatureProjector(
            num_dim=len(self.schema.notice.numeric),
            text_dim=768, num_proj_dim=128, text_proj_dim=128
        )
        self.company_projector = FeatureProjector(
            num_dim=len(self.schema.company.numeric),
            text_dim=768, num_proj_dim=128, text_proj_dim=128
        )
            
    def _build_id_mappings(self, notice_store: Dict, company_store: Dict):
        """ID 매핑 딕셔너리 생성"""
        if 'ids' not in notice_store:
            raise ValueError("notice_store에 'ids' 키가 없습니다. build_feature_store() 함수를 확인하세요.")
        
        if 'ids' not in company_store:
            raise ValueError("company_store에 'ids' 키가 없습니다. build_feature_store() 함수를 확인하세요.")
        
        notice_ids = notice_store['ids']
        self.notice_id_to_idx = {tuple(id_pair): idx for idx, id_pair in enumerate(notice_ids)}
        
        company_ids = company_store['ids']
        if company_ids:
            if isinstance(company_ids[0], (tuple, list)):
                self.company_id_to_idx = {str(id_tuple[0]): idx for idx, id_tuple in enumerate(company_ids)}
            else:
                self.company_id_to_idx = {str(company_id): idx for idx, company_id in enumerate(company_ids)}
    
    def _get_chunk_id(self, idx: int) -> int:
        """인덱스로부터 청크 ID 계산"""
        return idx // self.chunk_size
    
    def _load_chunk(self, chunk_id: int) -> pd.DataFrame:
        """청크 로딩 (스트리밍 모드)"""
        start_idx = chunk_id * self.chunk_size
        
        if self.is_train:
            query = f"""
            SELECT bidntceno, bidntceord, bizno, id
            FROM {self.schema.pair.table} 
            WHERE id >= {self.start_id} AND id <= {self.end_id}
            ORDER BY id 
            OFFSET {start_idx} LIMIT {self.chunk_size}
            """
        else:
            query = f"""
            SELECT bidntceno, bidntceord, bizno, id
            FROM {self.schema.pair.table} 
            WHERE id >= {self.start_id} AND id <= {self.end_id}
            ORDER BY id 
            OFFSET {start_idx} LIMIT {self.chunk_size}
            """
        
        return pd.read_sql(query, self.db_engine)
    
    def _get_pair_data(self, idx: int) -> pd.Series:
        """pair 데이터 조회 (streaming/static 모드 통합)"""
        if self.streaming:
            # 스트리밍 모드
            chunk_id = self._get_chunk_id(idx)
            
            if chunk_id not in self.chunk_cache:
                # 캐시 크기 제한
                if len(self.chunk_cache) >= 3:
                    oldest_chunk = min(self.chunk_cache.keys())
                    del self.chunk_cache[oldest_chunk]
                    if not self.load_all_features and oldest_chunk in self.feature_cache:
                        del self.feature_cache[oldest_chunk]
                
                # 새 청크 로딩
                chunk_df = self._load_chunk(chunk_id)
                self.chunk_cache[chunk_id] = chunk_df
                
                # 선택적 피처 로딩 (streaming + selective 모드)
                if not self.load_all_features:
                    notice_store, company_store = self._load_selective_features_for_chunk(chunk_df)
                    self.feature_cache[chunk_id] = {
                        'notice_store': notice_store,
                        'company_store': company_store,
                        'notice_id_to_idx': {tuple(id_pair): idx for idx, id_pair in enumerate(notice_store['ids'])},
                        'company_id_to_idx': {
                            str(id_tuple[0] if isinstance(id_tuple, (tuple, list)) else id_tuple): idx 
                            for idx, id_tuple in enumerate(company_store['ids'])
                        }
                    }
            
            # 청크 내 인덱스 계산
            chunk_df = self.chunk_cache[chunk_id]
            local_idx = idx % self.chunk_size
            
            if local_idx >= len(chunk_df):
                raise IndexError(f"Index {idx} out of range")
            
            return chunk_df.iloc[local_idx], chunk_id
        else:
            # 정적 모드
            return self.pairs.iloc[idx], None
    
    def __len__(self) -> int:
        return self.total_count
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """통합 배치 아이템 반환 - 성능 최적화"""
        if idx >= self.total_count:
            raise IndexError(f"Index {idx} out of range (total: {self.total_count})")
        
        # 하이브리드 모드 최적화: 피처가 미리 로드되어 있으면 Fast path 사용
        if self.load_all_features and hasattr(self, 'notice_store') and hasattr(self, 'company_store'):
            # Fast path for hybrid mode (load_all_features=True)
            # 피쳐는 메모리, 페어는 필요시 로드(청크 마다)
            if self.streaming:
                # 청크 체크 (실제로 새 청크일 때만 DB 엑세스)
                chunk_id = self._get_chunk_id(idx)
                if chunk_id not in self.chunk_cache:
                    chunk_df = self._load_chunk(chunk_id)
                    self.chunk_cache[chunk_id] = chunk_df
                    print(f"청크 {chunk_id} 로딩 완료") # 실제 청크 로드 확인용
                    
                # 청크 캐시에서 빠른 조회
                local_idx = idx % self.chunk_size
                pair = self.chunk_cache[chunk_id].iloc[local_idx]
                
            else:
                # 완전 static 모드
                pair = self.pairs.iloc[idx]
                
            bidntceno = pair['bidntceno']
            bidntceord = pair['bidntceord'] 
            bizno = str(pair['bizno'])
            
            # 미리 로드된 인덱스 맵에서 빠른 조회
            notice_key = (bidntceno, bidntceord)
            company_key = bizno
            
            notice_idx = self.notice_id_to_idx.get(notice_key, 0)
            company_idx = self.company_id_to_idx.get(company_key, 0)
            
            # 피처 추출 (미리 로드된 스토어 사용)
            notice_input = slice_and_convert_for_tower(
                store_result=self.notice_store,
                row_idx=[notice_idx],
                categorical_keys=self.schema.notice.categorical,
                text_cols=self.schema.notice.text,
                projector=self.notice_projector,
                fuse_projected=True
            )
            
            company_input = slice_and_convert_for_tower(
                store_result=self.company_store,
                row_idx=[company_idx], 
                categorical_keys=self.schema.company.categorical,
                text_cols=self.schema.company.text,
                projector=self.company_projector,
                fuse_projected=True
            )
            
            return {
                "notice": {"dense": notice_input["dense"], "kjt": notice_input["kjt"]},
                "company": {"dense": company_input["dense"], "kjt": company_input["kjt"]},
                "pair_info": {"bidntceno": bidntceno, "bidntceord": bidntceord, "bizno": bizno}
            }

        
        # 완전 스트리밍 모드 (load_all_features=False, streaming=True)      
        else:
            pair, chunk_id = self._get_pair_data(idx)
            bidntceno = pair['bidntceno']
            bidntceord = pair['bidntceord']
            bizno = str(pair['bizno'])
            
            # 피처 스토어 및 매핑 결정
            if self.load_all_features:
                notice_store = self.notice_store
                company_store = self.company_store
                notice_id_to_idx = self.notice_id_to_idx
                company_id_to_idx = self.company_id_to_idx
            else:
                feature_data = self.feature_cache[chunk_id]
                notice_store = feature_data['notice_store']
                company_store = feature_data['company_store']
                notice_id_to_idx = feature_data['notice_id_to_idx']
                company_id_to_idx = feature_data['company_id_to_idx']
            
            # 인덱스 찾기
            notice_key = (bidntceno, bidntceord)
            if notice_key not in notice_id_to_idx:
                print(f"WARNING: Notice ID {notice_key} not found, using index 0")
                notice_idx = 0
            else:
                notice_idx = notice_id_to_idx[notice_key]

            if bizno not in company_id_to_idx:
                print(f"WARNING: Company ID {bizno} not found, using index 0")
                company_idx = 0
            else:
                company_idx = company_id_to_idx[bizno]
            
            # 피처 추출
            notice_input = slice_and_convert_for_tower(
                store_result=notice_store,
                row_idx=[notice_idx],
                categorical_keys=self.schema.notice.categorical,
                text_cols=self.schema.notice.text,
                projector=self.notice_projector,
                fuse_projected=True
            )
            
            company_input = slice_and_convert_for_tower(
                store_result=company_store,
                row_idx=[company_idx],
                categorical_keys=self.schema.company.categorical,
                text_cols=self.schema.company.text,
                projector=self.company_projector,
                fuse_projected=True
            )
            
            return {
                "notice": {
                    "dense": notice_input["dense"],
                    "kjt": notice_input["kjt"]
                },
                "company": {
                    "dense": company_input["dense"],
                    "kjt": company_input["kjt"]
                },
                "pair_info": {
                    "bidntceno": bidntceno,
                    "bidntceord": bidntceord,
                    "bizno": bizno
                }
            }


def unified_bid_collate_fn(batch: List[Dict]) -> Dict[str, Dict[str, torch.Tensor]]:
    """통합 collate 함수"""
    batch_size = len(batch)
    
    notice_dense_list = []
    notice_kjt_list = []
    company_dense_list = []
    company_kjt_list = []
    
    for item in batch:
        notice_dense_list.append(item["notice"]["dense"])
        notice_kjt_list.append(item["notice"]["kjt"])
        company_dense_list.append(item["company"]["dense"])
        company_kjt_list.append(item["company"]["kjt"])
    
    notice_dense_batch = torch.cat(notice_dense_list, dim=0)
    company_dense_batch = torch.cat(company_dense_list, dim=0)
    
    # KJT 배치 결합
    from torchrec import KeyedJaggedTensor
    
    notice_values_list = []
    notice_lengths_list = []
    for kjt in notice_kjt_list:
        notice_values_list.append(kjt.values())
        notice_lengths_list.append(kjt.lengths())
    
    notice_values_batch = torch.cat(notice_values_list)
    notice_lengths_batch = torch.cat(notice_lengths_list)
    
    notice_kjt_batch = KeyedJaggedTensor.from_lengths_sync(
        keys=notice_kjt_list[0].keys(),
        values=notice_values_batch,
        lengths=notice_lengths_batch
    )
    
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


def create_unified_bid_dataloaders(
    db_engine: Engine,
    schema: TorchRecSchema,
    batch_size: int = 32,
    limit: Optional[int] = None,
    test_split: float = 0.1,
    shuffle_seed: int = 42,
    num_workers: int = 0,
    streaming: bool = False,
    chunk_size: int = 1000,
    load_all_features: bool = True,
    feature_chunksize: int = 5000,
    feature_limit: Optional[int] = None
    
) -> Tuple[DataLoader, DataLoader]:
    """
    통합 DataLoader 생성
    
    Args:
        streaming: True면 pair 스트리밍
        load_all_features: False면 선택적 피처 로딩
        chunk_size: 스트리밍 시 청크 크기
        feature_chunksize: 피처 로딩 시 청크 크기
        feature_limit: 피처 로딩 시 제한
    """
    
    # ===== 이 부분을 추가 =====
    shared_stores = None
    if load_all_features:
        print("공유 피처 데이터 로딩 중...")
        
        notice_store = build_feature_store(
            db_engine, schema.notice, 
            chunksize=feature_chunksize, 
            limit=feature_limit, 
            show_progress=True
        )
        
        company_store = build_feature_store(
            db_engine, schema.company, 
            chunksize=feature_chunksize, 
            limit=feature_limit, 
            show_progress=True
        )
        
        shared_stores = {
            'notice': notice_store,
            'company': company_store
        }
        print("공유 피처 스토어 생성 완료")
    # ========================
    
    train_dataset = UnifiedBidDataset(
        db_engine=db_engine, schema=schema, limit=limit,
        test_split=test_split, is_train=True, shuffle_seed=shuffle_seed,
        streaming=streaming, chunk_size=chunk_size, load_all_features=load_all_features,
        feature_chunksize=feature_chunksize, feature_limit=feature_limit,
        shared_feature_stores=shared_stores  
    )
    
    test_dataset = None
    if test_split > 0:
        test_dataset = UnifiedBidDataset(
            db_engine=db_engine, schema=schema, limit=limit,
            test_split=test_split, is_train=False, shuffle_seed=shuffle_seed,
            streaming=streaming, chunk_size=chunk_size, load_all_features=load_all_features,
            feature_chunksize=feature_chunksize, feature_limit=feature_limit,
            shared_feature_stores=shared_stores  
        )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=unified_bid_collate_fn, num_workers=num_workers, pin_memory=False
    )       
    
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=unified_bid_collate_fn, num_workers=num_workers, pin_memory=False
        )
    
    return train_loader, test_loader



if __name__ == "__main__":
    from data.database_connector import DatabaseConnector
    from src.torchrec_preprocess.schema import build_torchrec_schema_from_meta
    
    db = DatabaseConnector()
    engine = db.engine
    
    config = {
        "notice_table": "notice", "company_table": "company", 
        "pair_table": "bid_two_tower",
        "pair_notice_id_cols": ["bidntceno", "bidntceord"],
        "pair_company_id_cols": ["bizno"],
        "metadata_path": "meta/metadata.csv"
    }
    schema = build_torchrec_schema_from_meta(**config)
    
    print("=== 완전 스트리밍 모드 테스트 (권장) ===")
    train_loader, test_loader = create_unified_bid_dataloaders(
        db_engine=engine, schema=schema, batch_size=16,
        limit=10000, test_split=0.2, 
        streaming=True, load_all_features=False, chunk_size=500
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader) if test_loader else 'None'}")
    
    for batch_idx, batch in enumerate(train_loader):
        print(f"Batch {batch_idx}: Notice {batch['notice']['dense'].shape}, Company {batch['company']['dense'].shape}")
        if batch_idx >= 2:
            break