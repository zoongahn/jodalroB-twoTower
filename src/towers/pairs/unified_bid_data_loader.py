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
from src.torchrec_preprocess.feature_preprocessor import FeaturePreprocessor
from torchrec import KeyedJaggedTensor
from data.database_connector import DatabaseConnector


def _build_single_kjt(cat_single: torch.Tensor, keys: List[str]) -> KeyedJaggedTensor:
    """개별 샘플용 KJT 생성"""
    from torchrec import KeyedJaggedTensor
    assert len(cat_single.shape) == 1 and cat_single.shape[0] == len(keys)
    
    values = cat_single
    lengths = torch.ones(len(keys), dtype=torch.int32)
    
    return KeyedJaggedTensor.from_lengths_sync(
        keys=keys,
        values=values,
        lengths=lengths
    )

def _combine_kjts(kjt_list: List[KeyedJaggedTensor]) -> KeyedJaggedTensor:
    """KJT 리스트를 배치로 결합"""
    from torchrec import KeyedJaggedTensor
    
    if not kjt_list:
        raise ValueError("Empty KJT list")
    
    # 모든 KJT가 같은 키를 가져야 함
    keys = kjt_list[0].keys()
    
    # 모든 values와 lengths를 결합
    all_values = []
    all_lengths = []
    
    for kjt in kjt_list:
        all_values.append(kjt.values())
        all_lengths.append(kjt.lengths())
    
    combined_values = torch.cat(all_values)
    combined_lengths = torch.cat(all_lengths)
    
    return KeyedJaggedTensor.from_lengths_sync(
        keys=keys,
        values=combined_values,
        lengths=combined_lengths
    )

def _build_batch_kjt(cat_batch: torch.Tensor, keys: List[str]) -> KeyedJaggedTensor:
    """배치 단위로 KJT 생성 (벡터화)"""
    B, K = cat_batch.shape
    assert K == len(keys), f"Keys length {len(keys)} doesn't match tensor shape {K}"
    
    # 모든 값을 flatten
    values = cat_batch.reshape(-1)
    
    # 각 항목은 길이 1
    lengths = torch.ones(B * K, dtype=torch.int32)
    
    # 키를 B번 반복
    repeated_keys = keys * B
    
    return KeyedJaggedTensor.from_lengths_sync(
        keys=repeated_keys,
        values=values,
        lengths=lengths
    )


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
        self.preprocessed_stores = None  # 전처리된 스토어 저장용
        self._worker_id = None  # 워커 ID 추적
        
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
                # Preprocessed stores를 사용하는 경우
                if 'preprocessed' in shared_feature_stores:
                    print("  - Preprocessed 스토어 사용")
                    self.preprocessed_stores = shared_feature_stores['preprocessed']
                    self.notice_store = self.preprocessed_stores['notice']
                    self.company_store = self.preprocessed_stores['company']
                    self._build_id_mappings(self.notice_store, self.company_store)
                else:
                    # 기존 방식 (raw stores)
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
    
    def _ensure_db_connection(self):
        """워커별 DB 연결 확인 및 재생성"""
        # 워커 프로세스에서만 실행
        import torch.utils.data
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is not None:
            # 워커 프로세스인 경우
            if self._worker_id != worker_info.id:
                # 새 워커이거나 아직 연결이 없는 경우
                self._worker_id = worker_info.id
                
                # 기존 연결 종료
                if hasattr(self, 'db_engine'):
                    try:
                        self.db_engine.dispose()
                    except:
                        pass
                
                # 단일 프로세스 모드에서는 DB 재연결 불필요
                # print("단일 프로세스 모드: DB 재연결 생략")  # 로그 제거
    
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
    
    def _load_chunk(self, chunk_id: int) -> Dict[str, np.ndarray]:
        """청크 로딩 (스트리밍 모드) - NumPy 배열로 반환"""
        # DB 연결 확인 (워커별 연결)
        self._ensure_db_connection()
        
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
        
        chunk_df = pd.read_sql(query, self.db_engine)
        # DataFrame을 NumPy 배열로 변환하여 반환
        return {
            'bidntceno': chunk_df['bidntceno'].values,
            'bidntceord': chunk_df['bidntceord'].values,
            'bizno': chunk_df['bizno'].values,
            'id': chunk_df['id'].values
        }
    
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
                
                # 새 청크 로딩 (NumPy 배열)
                chunk_arrays = self._load_chunk(chunk_id)
                self.chunk_cache[chunk_id] = chunk_arrays
                
                # 선택적 피처 로딩 (streaming + selective 모드)
                if not self.load_all_features:
                    # chunk_arrays를 DataFrame으로 변환하여 전달
                    chunk_df = pd.DataFrame(chunk_arrays)
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
            
            # 청크 내 인덱스 계산 (NumPy 배열에서 직접 접근)
            chunk_arrays = self.chunk_cache[chunk_id]
            local_idx = idx % self.chunk_size
            
            if local_idx >= len(chunk_arrays['bidntceno']):
                raise IndexError(f"Index {idx} out of range")
            
            # NumPy 배열에서 직접 값 추출
            return {
                'bidntceno': chunk_arrays['bidntceno'][local_idx],
                'bidntceord': chunk_arrays['bidntceord'][local_idx],
                'bizno': chunk_arrays['bizno'][local_idx]
            }, chunk_id
        else:
            # 정적 모드
            return self.pairs.iloc[idx], None
    
    def __len__(self) -> int:
        return self.total_count
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """단순화된 아이템 반환 - load_all_features=True, streaming=True 전용"""
        if idx >= self.total_count:
            raise IndexError(f"Index {idx} out of range (total: {self.total_count})")
        
        # 청크에서 pair 데이터 조회
        chunk_id = self._get_chunk_id(idx)
        if chunk_id not in self.chunk_cache:
            chunk_arrays = self._load_chunk(chunk_id)
            self.chunk_cache[chunk_id] = chunk_arrays
            # print(f"청크 {chunk_id} 로딩 완료")  # 로그 제거 (워커별 중복 출력 방지)
            
        # 청크 캐시에서 빠른 조회
        local_idx = idx % self.chunk_size
        chunk_arrays = self.chunk_cache[chunk_id]
        bidntceno = chunk_arrays['bidntceno'][local_idx]
        bidntceord = chunk_arrays['bidntceord'][local_idx]
        bizno = str(chunk_arrays['bizno'][local_idx])
        
        # 인덱스 조회
        notice_key = (bidntceno, bidntceord)
        company_key = bizno
        
        notice_idx = self.notice_id_to_idx.get(notice_key, 0)
        company_idx = self.company_id_to_idx.get(company_key, 0)
        
        return {
            "notice_idx": notice_idx,
            "company_idx": company_idx,
            "pair_info": {"bidntceno": bidntceno, "bidntceord": bidntceord, "bizno": bizno}
        }


def create_collate_fn_original(dataset: 'UnifiedBidDataset'):
    """원본 collate 함수 (백업용) - load_all_features=True, streaming=True 전용"""
    from concurrent.futures import ThreadPoolExecutor
    
    def collate_fn(batch: List[Dict]) -> Dict[str, Dict[str, torch.Tensor]]:
        # 배치에서 인덱스들 추출
        notice_indices = [item["notice_idx"] for item in batch]
        company_indices = [item["company_idx"] for item in batch]
        
        # 멀티스레딩으로 notice와 company 병렬 처리
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Notice 처리 함수
            def process_notice():
                notice_dense = torch.from_numpy(
                    dataset.notice_store['dense_projected'][notice_indices]
                ).float()
                
                notice_cat = dataset.notice_store['categorical'][notice_indices]
                notice_kjt = _build_batch_kjt(
                    torch.from_numpy(notice_cat).long(),
                    dataset.schema.notice.categorical
                )
                return {"dense": notice_dense, "kjt": notice_kjt}
            
            # Company 처리 함수  
            def process_company():
                company_dense = torch.from_numpy(
                    dataset.company_store['dense_projected'][company_indices]
                ).float()
                
                company_cat = dataset.company_store['categorical'][company_indices]
                company_kjt = _build_batch_kjt(
                    torch.from_numpy(company_cat).long(),
                    dataset.schema.company.categorical
                )
                return {"dense": company_dense, "kjt": company_kjt}
            
            # 병렬 실행
            future_notice = executor.submit(process_notice)
            future_company = executor.submit(process_company)
            
            # 결과 수집
            notice_result = future_notice.result()
            company_result = future_company.result()
        
        return {
            "notice": notice_result,
            "company": company_result
        }
    
    return collate_fn


def create_collate_fn_gpu(dataset: 'UnifiedBidDataset'):
    """GPU 전용 collate 함수 (pin_memory=False 필요)"""
    from concurrent.futures import ThreadPoolExecutor
    
    def collate_fn_gpu(batch: List[Dict]) -> Dict[str, Dict[str, torch.Tensor]]:
        # 배치에서 인덱스들 추출
        notice_indices = [item["notice_idx"] for item in batch]
        company_indices = [item["company_idx"] for item in batch]
        
        # 멀티스레딩으로 notice와 company 병렬 처리
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Notice 처리 함수 (GPU 직접 전송)
            def process_notice():
                # Dense 데이터: GPU 직접 전송
                notice_dense = torch.from_numpy(
                    dataset.notice_store['dense_projected'][notice_indices]
                ).float().cuda()
                
                # Categorical 데이터: GPU에서 KJT 생성
                notice_cat = dataset.notice_store['categorical'][notice_indices]
                notice_kjt = _build_batch_kjt_gpu(
                    torch.from_numpy(notice_cat).long(),
                    dataset.schema.notice.categorical
                )
                return {"dense": notice_dense, "kjt": notice_kjt}
            
            # Company 처리 함수 (GPU 직접 전송)
            def process_company():
                # Dense 데이터: GPU 직접 전송
                company_dense = torch.from_numpy(
                    dataset.company_store['dense_projected'][company_indices]
                ).float().cuda()
                
                # Categorical 데이터: GPU에서 KJT 생성
                company_cat = dataset.company_store['categorical'][company_indices]
                company_kjt = _build_batch_kjt_gpu(
                    torch.from_numpy(company_cat).long(),
                    dataset.schema.company.categorical
                )
                return {"dense": company_dense, "kjt": company_kjt}
            
            # 병렬 실행
            future_notice = executor.submit(process_notice)
            future_company = executor.submit(process_company)
            
            # 결과 수집
            notice_result = future_notice.result()
            company_result = future_company.result()
        
        return {
            "notice": notice_result,
            "company": company_result
        }
    
    return collate_fn_gpu


def create_lightweight_collate_fn():
    """
    Lightweight collate function - 인덱스 추출만 담당
    AsyncBatchPreprocessor와 함께 사용
    """
    def lightweight_collate_fn(batch: List[Dict]) -> List[Dict]:
        # 단순히 배치 아이템들을 리스트로 반환
        # 무거운 전처리는 AsyncBatchPreprocessor에서 담당
        return batch
    
    return lightweight_collate_fn


def create_collate_fn(dataset: 'UnifiedBidDataset'):
    """GPU 최적화 collate 함수 - load_all_features=True, streaming=True 전용"""
    from concurrent.futures import ThreadPoolExecutor
    
    def collate_fn_gpu_optimized(batch: List[Dict]) -> Dict[str, Dict[str, torch.Tensor]]:
        # 배치에서 인덱스들 추출
        notice_indices = [item["notice_idx"] for item in batch]
        company_indices = [item["company_idx"] for item in batch]
        
        # 멀티스레딩으로 notice와 company 병렬 처리
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Notice 처리 함수 (GPU 최적화)
            def process_notice():
                # Dense 데이터: CPU에서 처리 후 나중에 GPU 이전
                notice_dense = torch.from_numpy(
                    dataset.notice_store['dense_projected'][notice_indices]
                ).float()
                
                # Categorical 데이터: CPU에서 KJT 생성 (pin_memory 호환)
                notice_cat = dataset.notice_store['categorical'][notice_indices]
                notice_kjt = _build_batch_kjt(
                    torch.from_numpy(notice_cat).long(),
                    dataset.schema.notice.categorical
                )
                return {"dense": notice_dense, "kjt": notice_kjt}
            
            # Company 처리 함수 (GPU 최적화)
            def process_company():
                # Dense 데이터: CPU에서 처리 후 나중에 GPU 이전  
                company_dense = torch.from_numpy(
                    dataset.company_store['dense_projected'][company_indices]
                ).float()
                
                # Categorical 데이터: CPU에서 KJT 생성 (pin_memory 호환)
                company_cat = dataset.company_store['categorical'][company_indices]
                company_kjt = _build_batch_kjt(
                    torch.from_numpy(company_cat).long(),
                    dataset.schema.company.categorical
                )
                return {"dense": company_dense, "kjt": company_kjt}
            
            # 병렬 실행
            future_notice = executor.submit(process_notice)
            future_company = executor.submit(process_company)
            
            # 결과 수집
            notice_result = future_notice.result()
            company_result = future_company.result()
        
        return {
            "notice": notice_result,
            "company": company_result
        }
    
    return collate_fn_gpu_optimized


def create_async_optimized_dataloaders(
    db_engine,
    schema,
    batch_size: int = 32,
    limit: Optional[int] = None,
    test_split: float = 0.1,
    shuffle_seed: int = 42,
    pin_memory: bool = True,
    streaming: bool = True,
    chunk_size: int = 1000,
    load_all_features: bool = True,
    feature_chunksize: int = 5000,
    feature_limit: Optional[int] = None,
    use_preprocessor: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    AsyncBatchPreprocessor를 사용하는 최적화된 DataLoader
    기존 create_unified_bid_dataloaders_gpu를 대체
    """
    from sqlalchemy.engine import Engine
    
    # 기존과 동일한 데이터 준비 로직
    shared_stores = None
    if load_all_features:
        if use_preprocessor and streaming:
            print("FeaturePreprocessor를 사용하여 피처 전처리 중...")
            from src.torchrec_preprocess.feature_preprocessor import FeaturePreprocessor
            
            preprocessor = FeaturePreprocessor(
                schema=schema,
                device='cuda:0' if torch.cuda.is_available() else 'cpu',
                num_proj_dim=128,
                text_proj_dim=128,
                batch_size=1024
            )
            
            preprocessed_stores = preprocessor.preprocess_all(
                db_engine=db_engine,
                feature_chunksize=feature_chunksize,
                feature_limit=feature_limit,
                show_progress=True
            )
            
            shared_stores = {
                'preprocessed': preprocessed_stores,
                'notice': preprocessed_stores['notice'],
                'company': preprocessed_stores['company']
            }
            print("Pre-projection 완료!")
    
    # 데이터셋 생성 (기존과 동일)
    train_dataset = UnifiedBidDataset(
        db_engine=db_engine, schema=schema, limit=limit, is_train=True, 
        test_split=test_split, shuffle_seed=shuffle_seed, streaming=streaming, 
        chunk_size=chunk_size, load_all_features=load_all_features,
        feature_chunksize=feature_chunksize, feature_limit=feature_limit,
        shared_feature_stores=shared_stores,
    )
    
    test_dataset = UnifiedBidDataset(
        db_engine=db_engine, schema=schema, limit=limit, is_train=False, 
        test_split=test_split, shuffle_seed=shuffle_seed, streaming=streaming, 
        chunk_size=chunk_size, load_all_features=load_all_features,
        feature_chunksize=feature_chunksize, feature_limit=feature_limit,
        shared_feature_stores=shared_stores,
    )
    
    # Lightweight collate 함수 사용
    lightweight_collate_fn = create_lightweight_collate_fn()
    
    # DataLoader (lightweight collate 함수 사용)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lightweight_collate_fn,
        num_workers=0,
        pin_memory=False  # AsyncBatchPreprocessor에서 pin_memory 처리
    )
    
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lightweight_collate_fn,
            num_workers=0,
            pin_memory=False
        )
    
    return train_loader, test_loader


# Helper 함수들
def _build_batch_kjt_original(categorical_data: torch.Tensor, categorical_keys: List[str]) -> 'KeyedJaggedTensor':
    """배치용 KJT 생성 (원본 백업)"""
    from torchrec import KeyedJaggedTensor
    
    batch_size, num_features = categorical_data.shape
    values_list = []
    lengths_list = []
    
    for i in range(num_features):
        feature_values = categorical_data[:, i]
        lengths = torch.ones(batch_size, dtype=torch.long)
        values_list.append(feature_values)
        lengths_list.append(lengths)
    
    all_values = torch.cat(values_list)
    all_lengths = torch.cat(lengths_list)
    
    return KeyedJaggedTensor.from_lengths_sync(
        keys=categorical_keys,
        values=all_values,
        lengths=all_lengths
    )


def _build_batch_kjt_gpu(categorical_data: torch.Tensor, categorical_keys: List[str]) -> 'KeyedJaggedTensor':
    """GPU 기반 배치용 KJT 생성 (최적화 버전)"""
    from torchrec import KeyedJaggedTensor
    
    # GPU로 이전
    if categorical_data.device.type != 'cuda':
        categorical_data = categorical_data.cuda()
    
    batch_size, num_features = categorical_data.shape
    
    # GPU에서 메모리 할당 및 reshape
    all_values = categorical_data.flatten()
    all_lengths = torch.ones(batch_size * num_features, dtype=torch.long, device=categorical_data.device)
    
    return KeyedJaggedTensor.from_lengths_sync(
        keys=categorical_keys,
        values=all_values,
        lengths=all_lengths
    )


def _build_batch_kjt(categorical_data: torch.Tensor, categorical_keys: List[str]) -> 'KeyedJaggedTensor':
    """배치용 KJT 생성 (CPU 버전 - 호환성 유지)"""
    from torchrec import KeyedJaggedTensor
    
    batch_size, num_features = categorical_data.shape
    
    # 최적화: 한 번에 메모리 할당 및 reshape
    all_values = categorical_data.flatten()
    all_lengths = torch.ones(batch_size * num_features, dtype=torch.long)
    
    return KeyedJaggedTensor.from_lengths_sync(
        keys=categorical_keys,
        values=all_values,
        lengths=all_lengths
    )


def _build_single_kjt(categorical_data: torch.Tensor, categorical_keys: List[str]) -> 'KeyedJaggedTensor':
    """단일 샘플용 KJT 생성"""
    from torchrec import KeyedJaggedTensor
    
    values = categorical_data.flatten()
    lengths = torch.ones(len(categorical_keys), dtype=torch.long)
    
    return KeyedJaggedTensor.from_lengths_sync(
        keys=categorical_keys,
        values=values,
        lengths=lengths
    )


def _combine_kjts(kjt_list: List['KeyedJaggedTensor']) -> 'KeyedJaggedTensor':
    """KJT 리스트를 배치로 결합"""
    from torchrec import KeyedJaggedTensor
    
    if not kjt_list:
        raise ValueError("Empty KJT list")
    
    keys = kjt_list[0].keys()
    values_list = [kjt.values() for kjt in kjt_list]
    lengths_list = [kjt.lengths() for kjt in kjt_list]
    
    combined_values = torch.cat(values_list)
    combined_lengths = torch.cat(lengths_list)
    
    return KeyedJaggedTensor.from_lengths_sync(
        keys=keys,
        values=combined_values,
        lengths=combined_lengths
    )



def create_unified_bid_dataloaders_gpu(
    db_engine: Engine,
    schema: TorchRecSchema,
    batch_size: int = 32,
    limit: Optional[int] = None,
    test_split: float = 0.1,
    shuffle_seed: int = 42,
    pin_memory: bool = False,  # GPU 버전에서는 False
    streaming: bool = False,
    chunk_size: int = 1000,
    load_all_features: bool = True,
    feature_chunksize: int = 5000,
    feature_limit: Optional[int] = None,
    use_preprocessor: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """GPU 최적화 DataLoader (옵션2: pin_memory=False + GPU 처리)"""
    
    # 기존 코드와 동일한 데이터 준비
    shared_stores = None
    if load_all_features:
        if use_preprocessor and streaming:
            print("FeaturePreprocessor를 사용하여 피처 전처리 중...")
            from src.torchrec_preprocess.feature_preprocessor import FeaturePreprocessor
            
            preprocessor = FeaturePreprocessor(
                schema=schema,
                device='cuda:0' if torch.cuda.is_available() else 'cpu',
                num_proj_dim=128,
                text_proj_dim=128,
                batch_size=1024
            )
            
            preprocessed_stores = preprocessor.preprocess_all(
                db_engine=db_engine,
                feature_chunksize=feature_chunksize,
                feature_limit=feature_limit,
                show_progress=True
            )
            
            shared_stores = {
                'preprocessed': preprocessed_stores,
                'notice': preprocessed_stores['notice'],
                'company': preprocessed_stores['company']
            }
            print("Pre-projection 완료!")
    
    # 데이터셋 생성
    train_dataset = UnifiedBidDataset(
        db_engine=db_engine, schema=schema, limit=limit, is_train=True, 
        test_split=test_split, shuffle_seed=shuffle_seed, streaming=streaming, 
        chunk_size=chunk_size, load_all_features=load_all_features,
        feature_chunksize=feature_chunksize, feature_limit=feature_limit,
        shared_feature_stores=shared_stores,
    )
    
    test_dataset = UnifiedBidDataset(
        db_engine=db_engine, schema=schema, limit=limit, is_train=False, 
        test_split=test_split, shuffle_seed=shuffle_seed, streaming=streaming, 
        chunk_size=chunk_size, load_all_features=load_all_features,
        feature_chunksize=feature_chunksize, feature_limit=feature_limit,
        shared_feature_stores=shared_stores,
    )
    
    # GPU collate 함수 사용
    train_collate_fn = create_collate_fn_gpu(train_dataset)
    
    # DataLoader (pin_memory=False 필수)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=train_collate_fn,
        num_workers=0,
        pin_memory=False  # GPU 버전에서는 False
    )
    
    test_loader = None
    if test_dataset is not None:
        test_collate_fn = create_collate_fn_gpu(test_dataset)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=test_collate_fn,
            num_workers=0,
            pin_memory=False  # GPU 버전에서는 False
        )
    
    return train_loader, test_loader


def create_unified_bid_dataloaders(
    db_engine: Engine,
    schema: TorchRecSchema,
    batch_size: int = 32,
    limit: Optional[int] = None,
    test_split: float = 0.1,
    shuffle_seed: int = 42,
    num_workers: int = 4,  # num_workers 추가
    pin_memory: bool = False,
    prefetch_factor: int = 2,  # prefetch_factor 추가
    persistent_workers: bool = False,  # persistent_workers 추가
    streaming: bool = False,
    chunk_size: int = 1000,
    load_all_features: bool = True,
    feature_chunksize: int = 5000,
    feature_limit: Optional[int] = None,
    use_preprocessor: bool = True  # Pre-projection 사용 여부
    
) -> Tuple[DataLoader, DataLoader]:
    """
    통합 DataLoader 생성
    
    Args:
        streaming: True면 pair 스트리밍
        load_all_features: False면 선택적 피처 로딩
        chunk_size: 스트리밍 시 청크 크기
        feature_chunksize: 피처 로딩 시 청크 크기
        feature_limit: 피처 로딩 시 제한
        use_preprocessor: True면 FeaturePreprocessor를 사용하여 pre-projection
    """
    
    shared_stores = None
    if load_all_features:
        if use_preprocessor and streaming:  # streaming=True, load_all_features=True에서만 사용
            print("FeaturePreprocessor를 사용하여 피처 전처리 중...")
            
            # FeaturePreprocessor로 전체 피처 전처리
            preprocessor = FeaturePreprocessor(
                schema=schema,
                device='cuda:0' if torch.cuda.is_available() else 'cpu',
                num_proj_dim=128,
                text_proj_dim=128,
                batch_size=1024
            )
            
            preprocessed_stores = preprocessor.preprocess_all(
                db_engine=db_engine,
                feature_chunksize=feature_chunksize,
                feature_limit=feature_limit,
                show_progress=True
            )
            
            # ID 매핑 생성
            notice_id_to_idx, company_id_to_idx = preprocessor.build_id_mappings(preprocessed_stores)
            
            shared_stores = {
                'preprocessed': preprocessed_stores,
                'notice': preprocessed_stores['notice'],
                'company': preprocessed_stores['company']
            }
            print("Pre-projection 완료!")
        else:
            # 기존 방식 (raw stores)
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
    
    # 단일 프로세스 모드: DB config 불필요
    
    train_dataset = UnifiedBidDataset(
        db_engine=db_engine, schema=schema, limit=limit,
        test_split=test_split, is_train=True, shuffle_seed=shuffle_seed,
        streaming=streaming, chunk_size=chunk_size, load_all_features=load_all_features,
        feature_chunksize=feature_chunksize, feature_limit=feature_limit,
        shared_feature_stores=shared_stores,
    )
    
    test_dataset = None
    if test_split > 0:
        test_dataset = UnifiedBidDataset(
            db_engine=db_engine, schema=schema, limit=limit,
            test_split=test_split, is_train=False, shuffle_seed=shuffle_seed,
            streaming=streaming, chunk_size=chunk_size, load_all_features=load_all_features,
            feature_chunksize=feature_chunksize, feature_limit=feature_limit,
            shared_feature_stores=shared_stores,
            )
    
    # 데이터셋에 맞는 collate 함수 생성
    train_collate_fn = create_collate_fn(train_dataset)
    
    # DataLoader 파라미터 (멀티프로세스 지원)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=train_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False
    )       
    
    test_loader = None
    if test_dataset is not None:
        test_collate_fn = create_collate_fn(test_dataset)
        
        # Test DataLoader (멀티프로세스 지원)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=test_collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers if num_workers > 0 else False
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