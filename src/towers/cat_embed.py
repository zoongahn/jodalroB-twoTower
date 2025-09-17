# src/cat_embed.py
from __future__ import annotations
from typing import Dict, List, Optional, Union
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torchrec import KeyedJaggedTensor


class CategoricalEmbedder(nn.Module):
    """
    기본 PyTorch 임베딩을 사용한 범주형 임베딩 클래스
    - 메타데이터 기반 vocab_size 자동 계산
    - 안전한 클램핑
    - TorchRec 의존성 제거
    """
    
    def __init__(
        self,
        keys: List[str],
        metadata_path: str | Path,
        table_name: str,
        embedding_dim: int = 64,
        device: Optional[str] = "cuda:0",
        safety_margin: int = 10,
    ):
        super().__init__()
        self.keys = keys
        self.embedding_dim = embedding_dim
        self.device_str = device or "cuda:0"
        self.safety_margin = safety_margin
        
        # 메타데이터에서 vocab_size 정보 추출
        self.vocab_sizes = self._extract_vocab_sizes(metadata_path, table_name, keys)
        
        print(f"[CategoricalEmbedder] Initializing with {len(keys)} features")
        
        # 기본 PyTorch Embedding 테이블들 생성
        self.embeddings = nn.ModuleDict()
        for key in self.keys:
            self.embeddings[key] = nn.Embedding(
                num_embeddings=self.vocab_sizes[key],
                embedding_dim=self.embedding_dim
            )
        
        # 전체 모듈을 지정된 디바이스로 이동
        self.to(torch.device(self.device_str))
        
    def _extract_vocab_sizes(
        self, 
        metadata_path: str | Path, 
        table_name: str, 
        keys: List[str]
    ) -> Dict[str, int]:
        """메타데이터에서 각 범주형 피처의 vocab_size 추출"""
        try:
            df_meta = pd.read_csv(metadata_path)
            df_table = df_meta[df_meta['테이블명'] == table_name].copy()
            
            vocab_sizes = {}
            for key in keys:
                col_info = df_table[df_table['컬럼명'] == key]
                
                if len(col_info) == 0:
                    print(f"[WARNING] Column '{key}' not found in metadata, using default vocab_size=1000")
                    vocab_sizes[key] = 1000
                    continue
                
                category_count = col_info['범주 갯수'].iloc[0]
                
                if pd.isna(category_count):
                    print(f"[WARNING] No category count info for '{key}', using default vocab_size=1000")
                    vocab_sizes[key] = 1000
                else:
                    vocab_size = int(category_count) + self.safety_margin
                    vocab_sizes[key] = vocab_size
                    # print(f"[INFO] {key}: category_count={int(category_count)} -> vocab_size={vocab_size}")
            
            return vocab_sizes
            
        except Exception as e:
            print(f"[ERROR] Failed to extract vocab_sizes from metadata: {e}")
            print(f"[FALLBACK] Using default vocab_size=1000 for all keys")
            return {key: 1000 for key in keys}
            
            
    def _kjt_to_dict(self, kjt: KeyedJaggedTensor) -> Dict[str, torch.Tensor]:
        if kjt is None:
            return {}
        
        values = kjt.values()
        lengths = kjt.lengths()
        keys = kjt.keys()
        
        # 배치 크기 계산
        unique_keys = len(self.keys)
        batch_size = len(values) // unique_keys
        
        result = {}
        
        # 각 고유 키별로 배치 데이터 수집
        for key_idx, key_name in enumerate(self.keys):
            batch_values = []
            
            # 배치 내 각 샘플에서 해당 키의 값 추출
            for batch_idx in range(batch_size):
                # KJT에서 이 키의 위치 계산
                kjt_key_idx = batch_idx * unique_keys + key_idx
                value_idx = kjt_key_idx  # lengths가 모두 1이므로 인덱스가 일치
                
                key_value = values[value_idx]
                
                # clamp 적용
                if key_name in self.vocab_sizes:
                    vocab_size = self.vocab_sizes[key_name]
                    key_value = torch.clamp(key_value, min=0, max=vocab_size - 1)
                
                batch_values.append(key_value)
            
            result[key_name] = torch.stack(batch_values)  # [B] 형태
        
        return result
    
    
    def forward(
        self, 
        kjt: Optional[KeyedJaggedTensor], 
        return_dict: bool = True
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Args:
            kjt: 범주형 피처들이 담긴 KeyedJaggedTensor
            return_dict: True면 Dict 반환, False면 concat된 텐서 반환
        
        Returns:
            return_dict=True: Dict[str, Tensor] - 각 피처별 임베딩
            return_dict=False: Tensor [B, total_embedding_dim] - 모든 임베딩 concat
        """
        

        
        if kjt is None:
            batch_size = 1
            device = torch.device(self.device_str)
            if return_dict:
                return {key: torch.zeros(batch_size, self.embedding_dim, device=device) 
                       for key in self.keys}
            else:
                return torch.zeros(batch_size, len(self.keys) * self.embedding_dim, device=device)
        
        try:
            # KJT를 일반 dict로 변환
            input_dict = self._kjt_to_dict(kjt)
            
            # 각 키별로 임베딩 수행
            embeddings = {}
            for key in self.keys:
                if key in input_dict and key in self.embeddings:
                    # 기본 PyTorch Embedding 사용
                    emb = self.embeddings[key](input_dict[key])  # [B, embedding_dim]
                    embeddings[key] = emb
                    # print(f"{key} embedding 성공: {emb.requires_grad}")

                else:
                    # 키가 없으면 영 벡터
                    batch_size = len(list(input_dict.values())[0]) if input_dict else 1
                    device = kjt.values().device if kjt else torch.device(self.device_str)
                    embeddings[key] = torch.zeros(batch_size, self.embedding_dim, device=device)
                    # print(f"{key} embedding 실패: {embeddings[key].requires_grad}")
            
            if return_dict:
                return embeddings
            else:
                # 모든 임베딩을 순서대로 concat
                tensors = [embeddings[key] for key in self.keys]
                result = torch.cat(tensors, dim=1)  # [B, total_embedding_dim]
                return result
                    
        except Exception as e:
            print(f"[ERROR] Embedding forward failed: {e}")
            # 에러 시 영 벡터 반환
            batch_size = 1
            device = kjt.values().device if kjt else torch.device(self.device_str)
            
            if return_dict:
                return {key: torch.zeros(batch_size, self.embedding_dim, device=device) 
                       for key in self.keys}
            else:
                return torch.zeros(batch_size, len(self.keys) * self.embedding_dim, device=device)


# 편의 함수
def create_categorical_embedder(
    keys: List[str],
    metadata_path: str | Path = "meta/metadata.csv",
    table_name: str = "notice",
    embedding_dim: int = 64,
    device: Optional[str] = "cuda:0",
) -> CategoricalEmbedder:
    """CategoricalEmbedder 생성을 위한 편의 함수"""
    return CategoricalEmbedder(
        keys=keys,
        metadata_path=metadata_path,
        table_name=table_name,
        embedding_dim=embedding_dim,
        device=device,
    )


if __name__ == "__main__":
    # 테스트용 코드
    from src.torchrec_preprocess.schema import build_torchrec_schema_from_meta
    
    # 스키마 로드
    schema = build_torchrec_schema_from_meta(
        notice_table="notice",
        company_table="company",
        pair_table="bid_two_tower",
        pair_notice_id_cols=["bidntceno", "bidntceord"],
        pair_company_id_cols=["bizno"],
        metadata_path="meta/metadata.csv",
    )
    
    # 테스트용 임베더 생성
    embedder = create_categorical_embedder(
        keys=schema.notice.categorical[:5],
        table_name="notice",
        embedding_dim=64,
        device="cuda:0",
    )
    
    print(f"\n[TEST] Created CategoricalEmbedder with keys: {embedder.keys}")
    print(f"[TEST] Vocab sizes: {embedder.vocab_sizes}")
    print(f"[TEST] Total embedding output dim: {len(embedder.keys) * embedder.embedding_dim}")