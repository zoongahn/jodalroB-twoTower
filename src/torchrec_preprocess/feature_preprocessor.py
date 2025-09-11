"""
Feature Preprocessor for Two-Tower Model
피처 전처리 파이프라인 관리 (projection, 캐싱 최적화 등)
"""
import torch
import numpy as np
from typing import Dict, Optional, Tuple
from tqdm import tqdm

from src.torchrec_preprocess.feature_store import build_feature_store
from src.torchrec_preprocess.feature_projector import FeatureProjector
from src.torchrec_preprocess.schema import TorchRecSchema, SideSchema


class FeaturePreprocessor:
    """
    피처 전처리 파이프라인 관리
    - Raw 데이터 로딩
    - Projection 적용 (GPU에서 배치 처리)
    - 범주형 데이터 준비
    - 캐싱 최적화
    """
    
    def __init__(
        self,
        schema: TorchRecSchema,
        device: str = 'cuda:0',
        num_proj_dim: int = 128,
        text_proj_dim: int = 128,
        batch_size: int = 1024  # Projection 시 배치 크기
    ):
        self.schema = schema
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_proj_dim = num_proj_dim
        self.text_proj_dim = text_proj_dim
        self.batch_size = batch_size
        
        # 각 타워별 projector 생성
        self._setup_projectors()
    
    def _setup_projectors(self):
        """각 타워별 Projector 설정"""
        self.projectors = {}
        
        # Notice projector
        if hasattr(self.schema, 'notice'):
            self.projectors['notice'] = FeatureProjector(
                num_dim=len(self.schema.notice.numeric),
                text_dim=768,
                num_proj_dim=self.num_proj_dim,
                text_proj_dim=self.text_proj_dim
            ).to(self.device)
        
        # Company projector
        if hasattr(self.schema, 'company'):
            self.projectors['company'] = FeatureProjector(
                num_dim=len(self.schema.company.numeric),
                text_dim=768,
                num_proj_dim=self.num_proj_dim,
                text_proj_dim=self.text_proj_dim
            ).to(self.device)
    
    def preprocess_all(
        self,
        db_engine,
        feature_chunksize: int = 5000,
        feature_limit: Optional[int] = None,
        show_progress: bool = True
    ) -> Dict[str, Dict]:
        """
        전체 피처 전처리 파이프라인 실행
        
        Returns:
            Dict[str, Dict]: {
                'notice': preprocessed_notice_store,
                'company': preprocessed_company_store
            }
        """
        preprocessed_stores = {}
        
        # Notice 피처 처리
        if hasattr(self.schema, 'notice'):
            print("Notice 피처 전처리 시작...")
            notice_store = self._preprocess_tower(
                db_engine=db_engine,
                tower_schema=self.schema.notice,
                tower_name='notice',
                chunksize=feature_chunksize,
                limit=feature_limit,
                show_progress=show_progress
            )
            preprocessed_stores['notice'] = notice_store
        
        # Company 피처 처리
        if hasattr(self.schema, 'company'):
            print("Company 피처 전처리 시작...")
            company_store = self._preprocess_tower(
                db_engine=db_engine,
                tower_schema=self.schema.company,
                tower_name='company',
                chunksize=feature_chunksize,
                limit=feature_limit,
                show_progress=show_progress
            )
            preprocessed_stores['company'] = company_store
        
        return preprocessed_stores
    
    def _preprocess_tower(
        self,
        db_engine,
        tower_schema: SideSchema,
        tower_name: str,
        chunksize: int,
        limit: Optional[int],
        show_progress: bool
    ) -> Dict:
        """
        단일 타워 전처리
        """
        # 1. Raw 피처 로딩
        print(f"  1. {tower_name} raw 데이터 로딩...")
        raw_store = build_feature_store(
            db_engine, tower_schema,
            chunksize=chunksize,
            limit=limit,
            show_progress=show_progress
        )
        
        # 2. Projection 적용
        print(f"  2. {tower_name} projection 적용...")
        projected_store = self._apply_projection(
            raw_store, 
            self.projectors[tower_name],
            tower_schema,
            tower_name
        )
        
        # 3. 범주형 데이터 준비 (이미 raw_store에 있음)
        print(f"  3. {tower_name} 범주형 데이터 준비...")
        projected_store['categorical'] = raw_store.get('categorical')
        projected_store['categorical_keys'] = tower_schema.categorical
        
        # 4. ID 매핑 보존
        projected_store['ids'] = raw_store.get('ids')
        
        print(f"  {tower_name} 전처리 완료!")
        return projected_store
    
    def _apply_projection(
        self,
        store: Dict,
        projector: FeatureProjector,
        tower_schema: SideSchema,
        tower_name: str
    ) -> Dict:
        """
        Projection을 배치 단위로 적용하여 CPU에 저장
        """
        # 원본 데이터 가져오기
        np_numeric = store.get('numeric')
        np_text = store.get('text') or {}
        
        n_samples = len(np_numeric) if np_numeric is not None else len(next(iter(np_text.values())))
        
        # 결과 저장용 리스트
        projected_batches = []
        
        # 배치 단위로 처리
        with torch.no_grad():
            for start_idx in tqdm(range(0, n_samples, self.batch_size), 
                                 desc=f"Projecting {tower_name}", 
                                 disable=not show_progress):
                end_idx = min(start_idx + self.batch_size, n_samples)
                
                # 배치 슬라이싱
                batch_numeric = None
                batch_text = {}
                
                if np_numeric is not None:
                    batch_numeric = torch.from_numpy(
                        np_numeric[start_idx:end_idx]
                    ).float().to(self.device)
                
                for col, mat in np_text.items():
                    batch_text[col] = torch.from_numpy(
                        mat[start_idx:end_idx]
                    ).float().to(self.device)
                
                # Projection 수행
                if batch_numeric is not None and batch_text:
                    proj_numeric, proj_text = projector(batch_numeric, batch_text)
                    
                    # 텍스트 컬럼들 concat
                    text_cols = tower_schema.text if tower_schema.text else list(proj_text.keys())
                    parts = [proj_numeric]
                    for col in text_cols:
                        if col in proj_text:
                            parts.append(proj_text[col])
                    
                    batch_projected = torch.cat(parts, dim=1)
                    
                elif batch_numeric is not None:
                    batch_projected, _ = projector(batch_numeric, {})
                    
                elif batch_text:
                    _, proj_text = projector(None, batch_text)
                    text_cols = tower_schema.text if tower_schema.text else list(proj_text.keys())
                    parts = []
                    for col in text_cols:
                        if col in proj_text:
                            parts.append(proj_text[col])
                    batch_projected = torch.cat(parts, dim=1) if parts else None
                else:
                    batch_projected = None
                
                # CPU로 이동하여 저장
                if batch_projected is not None:
                    projected_batches.append(batch_projected.cpu().numpy())
        
        # 전체 결과 합치기
        if projected_batches:
            dense_projected = np.concatenate(projected_batches, axis=0)
            print(f"    Projected shape: {dense_projected.shape}")
        else:
            dense_projected = None
            print(f"    No projection applied")
        
        # 결과 store 생성
        result_store = store.copy()
        result_store['dense_projected'] = dense_projected
        
        return result_store
    
    def build_id_mappings(self, stores: Dict[str, Dict]) -> Tuple[Dict, Dict]:
        """
        ID 매핑 딕셔너리 생성
        
        Returns:
            (notice_id_to_idx, company_id_to_idx)
        """
        notice_id_to_idx = {}
        company_id_to_idx = {}
        
        # Notice ID 매핑
        if 'notice' in stores:
            notice_ids = stores['notice'].get('ids', [])
            notice_id_to_idx = {
                tuple(id_pair): idx 
                for idx, id_pair in enumerate(notice_ids)
            }
        
        # Company ID 매핑
        if 'company' in stores:
            company_ids = stores['company'].get('ids', [])
            if company_ids:
                if isinstance(company_ids[0], (tuple, list)):
                    company_id_to_idx = {
                        str(id_tuple[0]): idx 
                        for idx, id_tuple in enumerate(company_ids)
                    }
                else:
                    company_id_to_idx = {
                        str(company_id): idx 
                        for idx, company_id in enumerate(company_ids)
                    }
        
        return notice_id_to_idx, company_id_to_idx


def show_progress(iterable, desc, total=None):
    """진행률 표시 헬퍼"""
    return tqdm(iterable, desc=desc, total=total)