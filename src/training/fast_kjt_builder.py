"""
고속 KJT 빌더 - 최적화된 KeyedJaggedTensor 생성
"""

import torch
import numpy as np
from typing import List, Dict, Optional
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class FastKJTBuilder:
    """
    KJT 생성 최적화
    - 벡터화 연산
    - 메모리 사전 할당
    - 불필요한 복사 제거
    """
    
    def __init__(self, categorical_keys: List[str], batch_size: int = 256):
        self.categorical_keys = categorical_keys
        self.batch_size = batch_size
        
        # 메모리 사전 할당
        self.preallocated_lengths = torch.zeros(
            len(categorical_keys) * batch_size, 
            dtype=torch.int32
        )
        
    def build_kjt_fast(
        self,
        categorical_data: np.ndarray,
        pin_memory: bool = True
    ) -> Optional[KeyedJaggedTensor]:
        """
        최적화된 KJT 생성
        
        Args:
            categorical_data: [batch_size, num_features] numpy array
            pin_memory: GPU 전송 최적화 여부
            
        Returns:
            KeyedJaggedTensor or None
        """
        if categorical_data is None or len(self.categorical_keys) == 0:
            return None
            
        batch_size = categorical_data.shape[0]
        num_features = len(self.categorical_keys)
        
        # 1. 벡터화된 방식으로 values와 lengths 계산
        # categorical_data가 이미 flatten된 형태라고 가정
        values = torch.from_numpy(categorical_data.ravel()).long()
        
        # 2. Lengths는 각 feature당 1개씩 (단순한 경우)
        lengths = torch.ones(batch_size * num_features, dtype=torch.int32)
        
        # 3. KJT 생성
        kjt = KeyedJaggedTensor(
            keys=self.categorical_keys,
            values=values,
            lengths=lengths
        )
        
        if pin_memory:
            kjt = kjt.pin_memory()
            
        return kjt


class OptimizedKJTBuilder:
    """
    더 복잡한 경우를 위한 최적화된 KJT 빌더
    """
    
    def __init__(self, schema):
        self.schema = schema
        
        # 각 피쳐의 최대 길이 사전 계산
        self.max_lengths = {}
        self.offsets_template = {}
        
    def build_kjt_optimized(
        self,
        indices: List[int],
        categorical_store: Dict,
        categorical_keys: List[str],
        pin_memory: bool = True
    ) -> Optional[KeyedJaggedTensor]:
        """
        스토어에서 직접 KJT 생성 (최적화)
        """
        if not categorical_keys or not categorical_store:
            return None
            
        batch_size = len(indices)
        
        # 모든 values와 lengths를 한번에 수집
        all_values = []
        all_lengths = []
        
        for key in categorical_keys:
            if key not in categorical_store:
                # 빈 피쳐
                all_lengths.extend([0] * batch_size)
                continue
                
            feature_data = categorical_store[key]
            
            # 벡터화된 인덱싱
            batch_values = []
            batch_lengths = []
            
            for idx in indices:
                data = feature_data[idx]
                if isinstance(data, (list, np.ndarray)):
                    batch_values.extend(data)
                    batch_lengths.append(len(data))
                else:
                    batch_values.append(data)
                    batch_lengths.append(1)
            
            all_values.extend(batch_values)
            all_lengths.extend(batch_lengths)
        
        # 텐서 생성
        if all_values:
            values_tensor = torch.tensor(all_values, dtype=torch.long)
            lengths_tensor = torch.tensor(all_lengths, dtype=torch.int32)
            
            kjt = KeyedJaggedTensor(
                keys=categorical_keys,
                values=values_tensor,
                lengths=lengths_tensor
            )
            
            if pin_memory:
                kjt = kjt.pin_memory()
                
            return kjt
            
        return None


def build_kjt_from_numpy(
    categorical_data: np.ndarray,
    categorical_keys: List[str],
    pin_memory: bool = True
) -> Optional[KeyedJaggedTensor]:
    """
    NumPy 배열에서 직접 KJT 생성 (가장 빠른 경로)
    """
    if categorical_data.size == 0:
        return None
        
    batch_size, num_features = categorical_data.shape
    
    # Flatten하고 텐서로 변환
    values = torch.from_numpy(categorical_data.ravel()).long()
    
    # 각 feature가 1개 값을 가진다고 가정
    lengths = torch.ones(batch_size * num_features, dtype=torch.int32)
    
    kjt = KeyedJaggedTensor(
        keys=categorical_keys,
        values=values,
        lengths=lengths
    )
    
    if pin_memory:
        kjt = kjt.pin_memory()
        
    return kjt