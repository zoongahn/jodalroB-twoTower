# src/torchrec_inputs.py
from __future__ import annotations
from typing import Dict, List, Sequence, Optional
import numpy as np
import torch
from torchrec import KeyedJaggedTensor
from sqlalchemy.engine import Engine

from src.torchrec_preprocess.feature_projector import FeatureProjector


from src.torchrec_preprocess.schema import build_torchrec_schema_from_meta, SideSchema
from src.torchrec_preprocess.feature_store import build_feature_store
from data.database_connector import DatabaseConnector


def _build_kjt_single(cat: torch.Tensor, keys: List[str]) -> Optional[KeyedJaggedTensor]:
    if cat is None:
        return None
    
    B, K = cat.shape
    assert K == len(keys)
    
    values = cat.reshape(-1)
    
    # 범주형 데이터는 정수형이어야 하고, gradient가 필요 없음
    # gradient는 임베딩 테이블 가중치에서 추적됨
    if values.requires_grad:
        values = values.detach()  # gradient 추적 제거
    
    lengths = torch.ones(B * K, dtype=torch.int32)
    repeated_keys = keys * B
    
    return KeyedJaggedTensor.from_lengths_sync(
        keys=repeated_keys,
        values=values,  # 정수형, requires_grad=False
        lengths=lengths
    )
    

def _concat_dense(numeric: Optional[torch.Tensor],
                  text_dict: Dict[str, torch.Tensor],
                  text_cols: Optional[List[str]]) -> Optional[torch.Tensor]:
    """
    numeric + 지정된 텍스트 컬럼들을 열 방향으로 concat → [B, D]
    """
    xs: List[torch.Tensor] = []
    if numeric is not None:
        xs.append(numeric.to(torch.float32))
    if text_cols:
        for col in text_cols:
            t = text_dict.get(col)
            if t is not None:
                xs.append(t.to(torch.float32))             # [B, 768]
    if not xs:
        return None
    return torch.cat(xs, dim=1)

def slice_and_convert_for_tower(
   store_result: Dict[str, object],
   row_idx: Optional[Sequence[int]] = None,
   *,
   categorical_keys: List[str],
   text_cols: Optional[List[str]] = None,
   return_text_separately: bool = False,
   projector: Optional[FeatureProjector] = None,
   fuse_projected: bool = True,
) -> Dict[str, object]:
   """
   FeatureStore 전체 매트릭스에서 row_idx로 배치 슬라이스 → TorchRec 입력으로 변환.
   
   Args:
       row_idx: 배치로 선택할 행 인덱스들. None이면 전체 데이터 사용.

   반환:
     {
       "dense": FloatTensor[B, D] or None,        # numeric + (선택) text concat
       "kjt": KeyedJaggedTensor or None,          # 범주형
       # return_text_separately=True면 아래도 포함
       "text": {col: FloatTensor[B, 768], ...}
     }
   """
   
   device = torch.device("cpu")
   
   # 1) numpy → batch slice (or full data)
   np_num  = store_result.get("numeric")
   np_cat  = store_result.get("categorical")
   np_text = store_result.get("text") or {}

   if row_idx is None:
       # 전체 데이터 사용
       num_batch = None if np_num is None else torch.from_numpy(np.asarray(np_num)).float()
       cat_batch = None if np_cat is None else torch.from_numpy(np.asarray(np_cat)).long()
       txt_batch: Dict[str, torch.Tensor] = {
           col: torch.from_numpy(np.asarray(mat)).float() for col, mat in np_text.items()
       } if np_text else {}
   else:
       # 지정된 행만 슬라이스
       num_batch = None if np_num is None else torch.from_numpy(np.asarray(np_num)[row_idx]).float()         # [B, Dn]
       cat_batch = None if np_cat is None else torch.from_numpy(np.asarray(np_cat)[row_idx]).long()          # [B, K]
       txt_batch: Dict[str, torch.Tensor] = {
           col: torch.from_numpy(np.asarray(mat)[row_idx]).float() for col, mat in np_text.items()
       } if np_text else {}

   # 2) Categorical -> KJT
   kjt  = _build_kjt_single(cat_batch, categorical_keys) if cat_batch is not None else None
   
   # 3) Projection 적용 (있다면)
   if projector is not None:
       num_proj, txt_proj = projector(num_batch, txt_batch)     # num→[B,128], 각 text→[B,128]
       # concat할 텍스트 컬럼 선택
       cols_to_concat = text_cols if text_cols else list(txt_proj.keys())
       if fuse_projected:
           xs = []
           if num_proj is not None:
               xs.append(num_proj)
           for c in cols_to_concat:
               if c in txt_proj:
                   xs.append(txt_proj[c])
           dense = torch.cat(xs, dim=1) if xs else None          # [B, 128 * (1 + |cols|)]
       else:
           # concat하지 않고 분리 반환
           dense = num_proj
           txt_batch = txt_proj
   else:
       # 기존 방식: numeric + (선택) text concat
       dense = _concat_dense(num_batch, txt_batch, text_cols=text_cols)

   out: Dict[str, object] = {"dense": dense, "kjt": kjt}
   if return_text_separately:
       out["text"] = txt_batch
   return out


# --- utils: move batch tensors to a device ---
def _move_batch_to_device(batch, device):
    if batch.get("dense") is not None:
        batch["dense"] = batch["dense"].to(device, non_blocking=True)
    if "text" in batch and batch["text"]:
        for k, v in batch["text"].items():
            batch["text"][k] = v.to(device, non_blocking=True)
    if batch.get("kjt") is not None:
        batch["kjt"] = batch["kjt"].to(device)
    return batch


def debug_tower_preprocess(
    store_result,
    row_idx,
    *,
    cat_keys,
    text_cols,
    num_dim,
    text_dim=768,
    num_proj_dim=128,
    text_proj_dim=128,
    device=None,  # ← 추가: 명시 안 하면 자동 선택
):
    
    # -----------------------------
    # 0) 디바이스 결정
    # -----------------------------
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- 원본(프로젝션 전) ---
    np_num  = store_result.get("numeric")
    np_cat  = store_result.get("categorical")
    np_text = store_result.get("text") or {}

    num_batch = None if np_num is None else torch.from_numpy(np_num[row_idx]).float()
    cat_batch = None if np_cat is None else torch.from_numpy(np_cat[row_idx]).long()
    txt_batch = {col: torch.from_numpy(mat[row_idx]).float() for col, mat in np_text.items()}
    


    print("\n=== [Before Projection] ===")
    print("numeric:", None if num_batch is None else tuple(num_batch.shape))
    print("categorical:", None if cat_batch is None else tuple(cat_batch.shape))
    if txt_batch:
        for col, t in txt_batch.items():
            print(f"text[{col}] (raw):", tuple(t.shape))  # (B, 768)

    # --- 프로젝터 준비 (수치/텍스트 별 투영 차원) ---
    projector = FeatureProjector(
        num_dim=num_dim,
        text_dim=text_dim,
        num_proj_dim=num_proj_dim,
        text_proj_dim=text_proj_dim,
    )

    # --- 프로젝션 + (옵션) 결합 ---
    batch = slice_and_convert_for_tower(
        store_result=store_result,
        row_idx=row_idx,
        categorical_keys=cat_keys,
        text_cols=text_cols,             # concat 대상 텍스트 컬럼들
        return_text_separately=True,     # 프로젝션 결과 확인용
        projector=projector,             # ← 적용
        fuse_projected=True,             # ← 결합(Concat)
    )

    # -----------------------------
    # 1) 디바이스 통일
    # -----------------------------
    batch = _move_batch_to_device(batch, device)

    # --- 프로젝션 결과(결합 전) ---
    print("\n=== [After Projection, Before Fusion] ===")
    if "text" in batch and batch["text"]:
        for col, t in batch["text"].items():
            print(f"text[{col}] (proj {text_proj_dim}):", tuple(t.shape))  # (B, text_proj_dim)

    # --- 최종 결합 결과(수치+텍스트) ---
    print("\n=== [Fused Dense] ===")
    fused = batch["dense"]  # 이미 device 통일됨
    print("dense (fused):", None if fused is None else tuple(fused.shape))
    if fused is not None:
        num_part = num_proj_dim if num_batch is not None else 0
        txt_count = len(text_cols) if text_cols else 0
        txt_part = text_proj_dim * txt_count
        print(f" → breakdown: num={num_part}, text_total={txt_part} ({text_proj_dim}×{txt_count}), sum={num_part + txt_part}")
        print("dense stats: min=", fused.min().item(), "max=", fused.max().item(), "mean=", fused.mean().item())
    
    from src.towers.cat_embed import SafeCategoricalEmbedder
    embedder = SafeCategoricalEmbedder(
        keys=cat_keys,
        metadata_path="meta/metadata.csv",
        table_name="notice",
        embedding_dim=64,
        device=str(device),
    )

    kjt = batch["kjt"]
    cat_dense = None
    print("\n=== [Categorical KJT] ===")
    if kjt is not None:
        print("kjt values:", tuple(kjt.values().shape), "lengths:", tuple(kjt.lengths().shape))
        print("kjt keys (head):", cat_keys[: min(8, len(cat_keys))], " ...")

    # 임베딩 전에 디버그 출력
    # forward()가 내부적으로 clamp/shift & device 정렬을 수행
    cat_dense = embedder(kjt, return_dict=False)  # [B, K*E]
        
    print("\n=== [Categorical Embedding] ===")
    print("cat_dense:", tuple(cat_dense.shape))

    # --- 최종 결합 (num+text + cat) ---
    print("\n=== [Final fused: num+text+cat] ===")
    if fused is None and cat_dense is None:
        print("final: None")
    else:
        parts = []
        if isinstance(fused, torch.Tensor):
            parts.append(fused)
        if isinstance(cat_dense, torch.Tensor):
            parts.append(cat_dense)

        # 모든 파트를 동일 device로 보장
        parts = [p.to(device, non_blocking=True) for p in parts]
        if fused is not None:
            fused = fused.to(cat_dense.device)
        final = torch.cat(parts, dim=1) if fused is not None else cat_dense
        print("final:", tuple(final.shape))
        
        
def print_tower_input_summary(batch, categorical_key_count: int):
    print(f"\n=== Tower Input Summary ===")
    
    # Dense 정보
    if batch.get('dense') is not None:
        dense = batch['dense']
        print(f"Dense: {tuple(dense.shape)} | min={dense.min():.4f} max={dense.max():.4f} mean={dense.mean():.4f}")
        print(f"       Device: {dense.device} | Dtype: {dense.dtype} | Grad: {dense.requires_grad}")
    else:
        print("Dense: None")
    
    # KJT 정보
    if batch.get('kjt') is not None:
        kjt = batch['kjt']
        print(f"KJT: {len(kjt.keys())} keys | values={tuple(kjt.values().shape)} | lengths={tuple(kjt.lengths().shape)}")
        print(f"     Keys: {kjt.keys()[:categorical_key_count]}")
        for i in range(5):
            print(f"     Values {i}: {kjt.values()[i*categorical_key_count:(i+1)*categorical_key_count].tolist()}")
        print(f"     Device: {kjt.device()}")
    else:
        print("KJT: None")
    
    # Text 정보 (있다면)
    if batch.get('text'):
        print("Text:")
        for col, tensor in batch['text'].items():
            print(f"  {col}: {tuple(tensor.shape)} | min={tensor.min():.4f} max={tensor.max():.4f}")
    else:
        print("Text: None")
    
    print("=" * 50)


def get_tower_input(db_engine: Engine, schema: SideSchema, tower_name: str, chunksize: int, limit: int, show_progress: bool = True):
    print(f"{tower_name} 타워 입력 생성 시작 ------------------------------")
    result = build_feature_store(db_engine, schema, chunksize=chunksize, limit=limit, show_progress=show_progress)
    print(f"{tower_name} feature store 생성 완료 ------------------------------")
    
    projector = FeatureProjector(num_dim=len(schema.numeric), text_dim=768, num_proj_dim=128, text_proj_dim=128)
    print(f"{tower_name} projector 생성 완료 ------------------------------")
    
    # 전체 row에 대해서 수행
    tower_batch = slice_and_convert_for_tower(
        store_result=result,
        categorical_keys=schema.categorical,
        text_cols=schema.text,
        return_text_separately=True,
        projector=projector,
        fuse_projected=True,
    )
    print(f"{tower_name} 타워 입력 생성 완료 ------------------------------")
    
        # 모든 텐서를 cuda:0으로 이동
    device = torch.device("cuda:0")
    if tower_batch["dense"] is not None:
        tower_batch["dense"] = tower_batch["dense"].to(device)
    if tower_batch["kjt"] is not None:
        tower_batch["kjt"] = tower_batch["kjt"].to(device)
    if "text" in tower_batch and tower_batch["text"]:
        for key, tensor in tower_batch["text"].items():
            tower_batch["text"][key] = tensor.to(device)
    
    return tower_batch




def preprocess_all(db_engine: Engine, schema_config: dict=None):
    print("device:", device)
    
    if schema_config is None:
        schema_config = {
            "notice_table": "notice",
            "company_table": "company", 
            "pair_table": "bid_two_tower",
            "pair_notice_id_cols": ["bidntceno", "bidntceord"],
            "pair_company_id_cols": ["bizno"],
            "metadata_path": "meta/metadata.csv"
        }

    print("3개 테이블 스키마 생성 시작 ------------------------------")
    schema = build_torchrec_schema_from_meta(**schema_config)
    print("3개 테이블 스키마 생성 완료 ------------------------------")
    
    # NOTICE -------------------------------------------------
    notice_schema = schema.notice
    notice_batch = get_tower_input(db_engine, notice_schema, "notice", chunksize=1000, limit=10000, show_progress=True)
    
    # 결과 출력
    print_tower_input_summary(notice_batch, len(notice_schema.categorical))
    
    # COMPANY -------------------------------------------------
    company_schema = schema.company
    company_batch = get_tower_input(db_engine, company_schema, "company", chunksize=1000, limit=10000, show_progress=True)
    
    # 결과 출력
    print_tower_input_summary(company_batch, len(company_schema.categorical))
    
    
    

if __name__ == "__main__":
    db = DatabaseConnector()
    engine = db.engine
    
    preprocess_all(engine)
    
    

# if __name__ == "__main__":
#     db = DatabaseConnector()
#     engine = db.engine
    
#     schema = build_torchrec_schema_from_meta(
#         notice_table="notice",
#         company_table="company",
#         pair_table="bid_two_tower",
#         pair_notice_id_cols=["bidntceno", "bidntceord"],
#         pair_company_id_cols=["bizno"],
#         metadata_path="meta/metadata.csv",
#     )
    
#     notice_schema = schema.notice
    
#     result = build_feature_store(engine, notice_schema, chunksize=1000, limit=10000)
    
    
#     notice_proj = FeatureProjector(num_dim=len(notice_schema.numeric), text_dim=768, num_proj_dim=128, text_proj_dim=128)
    
#     # 배치로 뽑을 row 인덱스 예시 (limit=10000 기준)
#     row_idx = [0, 1, 2, 10, 99, 123, 222]  # 맨 끝 포함해서 검증

#     notice_batch = slice_and_convert_for_tower(
#         store_result=result,                          # build_feature_store 에서 받은 dict
#         row_idx=row_idx,
#         categorical_keys=schema.notice.categorical,   # 메타에서 온 카테고리 키 순서
#         text_cols=schema.notice.text,                 # 텍스트 컬럼들 (pgvector 컬럼명 리스트)
#         return_text_separately=True,                  # 텍스트를 분리해서도 확인하려면 True
#         projector=notice_proj,
#         fuse_projected=True,
#     )

#     debug_tower_preprocess(
#         result, 
#         row_idx, 
#         cat_keys=schema.notice.categorical, 
#         text_cols=schema.notice.text, 
#         num_dim=len(notice_schema.numeric),
#         text_dim=768,
#         num_proj_dim=128,
#         text_proj_dim=128)