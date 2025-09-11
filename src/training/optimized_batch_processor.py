"""
최적화된 배치 프로세서 - KJT 생성 최적화에 집중
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
import time
import logging


class OptimizedBatchProcessor:
    """
    배치 처리 최적화 버전
    - Vectorized 연산 활용
    - 불필요한 복사 제거
    - 효율적인 메모리 액세스 패턴
    """
    
    def __init__(self, dataset, pin_memory: bool = True):
        self.dataset = dataset
        self.pin_memory = pin_memory
        self.logger = logging.getLogger(__name__)
        
        # 미리 계산 가능한 값들 캐싱
        self._prepare_optimizations()
        
    def _prepare_optimizations(self):
        """최적화를 위한 사전 준비"""
        # 1. 카테고리컬 피쳐의 offset 미리 계산
        self.notice_cat_offsets = {}
        self.company_cat_offsets = {}
        
        # Notice 카테고리컬 오프셋
        for key in self.dataset.schema.notice.categorical:
            if key in self.dataset.notice_store["categorical"]:
                data = self.dataset.notice_store["categorical"][key]
                # 각 샘플의 시작 위치 미리 계산
                lengths = np.array([len(data[i]) for i in range(len(data))])
                offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
                offsets[1:] = np.cumsum(lengths)
                self.notice_cat_offsets[key] = offsets
                
        # Company 카테고리컬 오프셋
        for key in self.dataset.schema.company.categorical:
            if key in self.dataset.company_store["categorical"]:
                data = self.dataset.company_store["categorical"][key]
                lengths = np.array([len(data[i]) for i in range(len(data))])
                offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
                offsets[1:] = np.cumsum(lengths)
                self.company_cat_offsets[key] = offsets
    
    def process_batch_optimized(self, raw_batch: List[Dict]) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        최적화된 배치 처리
        """
        start_time = time.time()
        
        # 1. 벡터화된 인덱스 추출
        batch_size = len(raw_batch)
        notice_indices = np.array([item["notice_idx"] for item in raw_batch], dtype=np.int32)
        company_indices = np.array([item["company_idx"] for item in raw_batch], dtype=np.int32)
        
        # 2. Notice 데이터 처리 (최적화)
        notice_data = self._process_tower_optimized(
            indices=notice_indices,
            store=self.dataset.notice_store,
            categorical_keys=self.dataset.schema.notice.categorical,
            cat_offsets=self.notice_cat_offsets,
            tower_name="notice"
        )
        
        # 3. Company 데이터 처리 (최적화)
        company_data = self._process_tower_optimized(
            indices=company_indices,
            store=self.dataset.company_store,
            categorical_keys=self.dataset.schema.company.categorical,
            cat_offsets=self.company_cat_offsets,
            tower_name="company"
        )
        
        process_time = time.time() - start_time
        
        return {
            "notice": notice_data,
            "company": company_data,
            "_meta": {
                "process_time": process_time,
                "batch_size": batch_size,
                "optimized": True
            }
        }
    
    def _process_tower_optimized(
        self, 
        indices: np.ndarray,
        store: Dict,
        categorical_keys: List[str],
        cat_offsets: Dict,
        tower_name: str
    ) -> Dict[str, torch.Tensor]:
        """
        최적화된 타워 데이터 처리
        """
        batch_size = len(indices)
        
        # 1. Dense 데이터 - 벡터화된 인덱싱
        if "dense" in store and store["dense"] is not None:
            # NumPy의 advanced indexing 활용
            dense_batch = store["dense"][indices]
            dense_tensor = torch.from_numpy(dense_batch).float()
            if self.pin_memory:
                dense_tensor = dense_tensor.pin_memory()
        else:
            dense_tensor = torch.zeros((batch_size, 0), dtype=torch.float32)
            if self.pin_memory:
                dense_tensor = dense_tensor.pin_memory()
        
        # 2. Categorical 데이터 - 최적화된 KJT 생성
        kjt = self._create_kjt_optimized(
            indices=indices,
            store=store,
            categorical_keys=categorical_keys,
            cat_offsets=cat_offsets
        )
        
        return {
            "dense": dense_tensor,
            "categorical": kjt
        }
    
    def _create_kjt_optimized(
        self,
        indices: np.ndarray,
        store: Dict,
        categorical_keys: List[str],
        cat_offsets: Dict
    ) -> Optional[KeyedJaggedTensor]:
        """
        최적화된 KJT 생성
        - 사전 계산된 오프셋 활용
        - 벡터화된 연산
        - 메모리 복사 최소화
        """
        if "categorical" not in store or not categorical_keys:
            return None
        
        batch_size = len(indices)
        cat_store = store["categorical"]
        
        # 모든 카테고리컬 피쳐를 한번에 처리
        all_values = []
        all_lengths = []
        
        for key in categorical_keys:
            if key not in cat_store:
                # 빈 피쳐 처리
                all_values.append(np.array([], dtype=np.int64))
                all_lengths.append(np.zeros(batch_size, dtype=np.int32))
                continue
            
            feature_data = cat_store[key]
            
            # 벡터화된 방식으로 배치 데이터 수집
            batch_values = []
            batch_lengths = np.zeros(batch_size, dtype=np.int32)
            
            for i, idx in enumerate(indices):
                values = feature_data[idx]
                if len(values) > 0:
                    batch_values.extend(values)
                    batch_lengths[i] = len(values)
            
            all_values.append(np.array(batch_values, dtype=np.int64))
            all_lengths.append(batch_lengths)
        
        # 통합된 텐서 생성
        if all_values:
            # Concatenate all values
            values_concat = np.concatenate([v for v in all_values if len(v) > 0])
            values_tensor = torch.from_numpy(values_concat)
            
            # Create lengths tensor
            lengths_concat = np.concatenate(all_lengths)
            lengths_tensor = torch.from_numpy(lengths_concat)
            
            # KJT 생성
            kjt = KeyedJaggedTensor(
                keys=categorical_keys,
                values=values_tensor,
                lengths=lengths_tensor
            )
            
            if self.pin_memory:
                kjt = kjt.pin_memory()
                
            return kjt
        
        return None


class AsyncOptimizedProcessor:
    """
    비동기 + 최적화 조합 프로세서
    """
    
    def __init__(
        self,
        dataset,
        num_workers: int = 16,
        use_multiprocessing: bool = False,
        pin_memory: bool = True
    ):
        self.dataset = dataset
        self.num_workers = num_workers
        self.use_multiprocessing = use_multiprocessing
        self.pin_memory = pin_memory
        
        # 최적화된 프로세서 생성
        self.processor = OptimizedBatchProcessor(dataset, pin_memory)
        
        if use_multiprocessing:
            from concurrent.futures import ProcessPoolExecutor
            self.executor = ProcessPoolExecutor(max_workers=num_workers)
        else:
            from concurrent.futures import ThreadPoolExecutor
            self.executor = ThreadPoolExecutor(max_workers=num_workers)
            
        self.logger = logging.getLogger(__name__)
        
    def process_batch_async(self, raw_batch: List[Dict]):
        """비동기 배치 처리 제출"""
        return self.executor.submit(
            self.processor.process_batch_optimized,
            raw_batch
        )
    
    def shutdown(self):
        """Executor 종료"""
        self.executor.shutdown(wait=True)