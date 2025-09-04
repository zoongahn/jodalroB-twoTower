#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
text_preprocess.py
- Hugging Face transformers(Tokenizer+AutoModel)만 사용해서 문장 임베딩 생성
- mean pooling(+attention mask)으로 문장 벡터화
- PK 보존: notice(bidntceno, bidntceord), company(bizno)
- 모델명은 .env의 TEXT_EMBEDDING_MODEL 사용(없으면 koELECTRA base)
- config의 컬럼 옵션: use, max_length, lowercase, strip, drop_if_empty, fillna, l2_normalize, batch_size
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

# .env 사용 (없으면 무시하려면 python-dotenv 설치 필수)
from dotenv import load_dotenv
load_dotenv()

# -----------------------------
# 기본 매핑/유틸
# -----------------------------

TABLE_PK_MAP: Dict[str, List[str]] = {
    "notice":  ["bidntceno", "bidntceord"],
    "company": ["bizno"],
}

DEFAULT_MODEL_NAME = os.getenv(
    "TEXT_EMBEDDING_MODEL",
    "monologg/koelectra-base-v3-discriminator"
)

def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n

# -----------------------------
# 컬럼 설정
# -----------------------------

@dataclass
class TextColumnConfig:
    name: str
    max_length: Optional[int] = 128
    lowercase: bool = True
    strip: bool = True
    drop_if_empty: bool = False
    fillna: str = ""
    l2_normalize: bool = True
    batch_size: int = 64

    def clean(self, s: pd.Series) -> pd.Series:
        x = s.astype("string").fillna(self.fillna)
        if self.strip:
            x = x.str.strip()
        if self.lowercase:
            x = x.str.lower()
        return x

# -----------------------------
# 전처리기
# -----------------------------

@dataclass
class TextPreprocessor:
    config: Dict[str, Dict]
    table_pk_map: Dict[str, List[str]] = field(default_factory=lambda: TABLE_PK_MAP.copy())
    default_model_name: str = DEFAULT_MODEL_NAME

    # (tokenizer, model) 캐시: 같은 모델/길이 조합 재사용
    _backend_cache: Dict[str, Tuple[AutoTokenizer, AutoModel]] = field(default_factory=dict, init=False, repr=False)
    _device: torch.device = field(default_factory=_device, init=False, repr=False)

    def _get_backend(self, model_name: Optional[str], max_length: Optional[int]) -> Tuple[AutoTokenizer, AutoModel, int]:
        """
        모델/토크나이저 로드 + 캐시
        캐시키: "<model_name>::len=<max_length or auto>"
        """
        name = model_name or self.default_model_name
        key = f"{name}::len={int(max_length) if max_length else 'auto'}"

        if key not in self._backend_cache:
            tok = AutoTokenizer.from_pretrained(name)
            mdl = AutoModel.from_pretrained(name)
            mdl.eval().to(self._device)
            self._backend_cache[key] = (tok, mdl)

        tok, mdl = self._backend_cache[key]
        hidden_size = mdl.config.hidden_size
        return tok, mdl, hidden_size

    @torch.inference_mode()
    def _encode_texts(
        self,
        texts: List[str],
        tokenizer: AutoTokenizer,
        model: AutoModel,
        max_length: Optional[int],
        batch_size: int
    ) -> np.ndarray:
        """
        mean pooling with attention mask:
        emb = sum(hidden * mask) / sum(mask)
        """
        bs = max(1, int(batch_size))
        max_len = int(max_length) if max_length else None
        device = self._device

        outs = []
        for i in range(0, len(texts), bs):
            batch = texts[i:i+bs]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt"
            )

            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            output = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = output.last_hidden_state  # [B, L, H]

            # mean pooling (mask aware)
            mask = attention_mask.unsqueeze(-1).type_as(last_hidden)  # [B,L,1]
            summed = (last_hidden * mask).sum(dim=1)                  # [B,H]
            counts = mask.sum(dim=1).clamp(min=1e-6)                  # [B,1]
            embs = (summed / counts).cpu().numpy().astype("float32")  # [B,H]

            outs.append(embs)

        return np.vstack(outs) if outs else np.zeros((0, model.config.hidden_size), dtype="float32")

    def fit(self, df: pd.DataFrame, table_name: str) -> "TextPreprocessor":
        # 텍스트 전처리는 별도 통계 없음
        return self

    def transform(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        original_order = list(df.columns)

        # PK 보존
        pk_all = self.table_pk_map.get(table_name, []) or []
        missing = [c for c in pk_all if c not in df.columns]
        if missing:
            raise KeyError(f"[{table_name}] PK 컬럼 누락: {missing}")
        out_dict: Dict[str, pd.Series] = {pk: df[pk] for pk in pk_all}

        # 컬럼별 임베딩
        for col, raw_cfg in self.config.items():
            if col not in df.columns:
                continue
            if not raw_cfg.get("use", True):
                continue

            # normalize -> l2_normalize 매핑 + 허용키 필터
            allowed = {"max_length", "lowercase", "strip", "drop_if_empty", "fillna", "l2_normalize", "batch_size"}
            raw = dict(raw_cfg)
            if "normalize" in raw and "l2_normalize" not in raw:
                raw["l2_normalize"] = raw.pop("normalize")
            tcfg_kwargs = {k: v for k, v in raw.items() if k in allowed}
            tcfg = TextColumnConfig(name=col, **tcfg_kwargs)

            # 텍스트 클린
            series_clean = tcfg.clean(df[col])
            texts = series_clean.tolist()

            # 백엔드 로드(모델명은 .env에서 통일 사용)
            tokenizer, model, hidden_size = self._get_backend(None, tcfg.max_length)

            # 임베딩
            embs = self._encode_texts(texts, tokenizer, model, tcfg.max_length, tcfg.batch_size)  # [N, H]
            if tcfg.l2_normalize:
                embs = _l2_normalize(embs)

            # 열 확장
            for d in range(embs.shape[1]):
                col_name = f"{col}_emb{d:03d}"
                out_dict[col_name] = pd.Series(embs[:, d], index=df.index, dtype="float32", name=col_name)

        # 원래 순서 우선 + 새 열 뒤에
        out_df = pd.DataFrame({c: out_dict[c] for c in original_order if c in out_dict})
        extras = [c for c in out_dict if c not in out_df.columns]

        if extras:
            # Series들을 한 번에 합쳐 DataFrame으로 만들고 concat
            extra_df = pd.concat([out_dict[c] for c in extras], axis=1)
            out_df = pd.concat([out_df, extra_df], axis=1)  # ← 한 번에 병합 (경고 해결)

        return out_df

# -----------------------------
# 외부에서 호출할 래퍼
# -----------------------------

def preprocess_text_data(df: pd.DataFrame, table_name: str, json_config_path: str, pre: Optional[TextPreprocessor] = None) -> pd.DataFrame:
    cfg = load_config(json_config_path)
    pk_cols = [c for c in TABLE_PK_MAP.get(table_name, []) if c in df.columns]
    text_cols = [c for c in cfg.keys() if c in df.columns]
    use_cols = pk_cols + [c for c in text_cols if c not in pk_cols]
    work = df[use_cols].copy()

    if pre is None:
        pre = TextPreprocessor(cfg, table_pk_map=TABLE_PK_MAP)

    pre.fit(work, table_name=table_name)
    out = pre.transform(work, table_name=table_name)
    return out

# -----------------------------
# 배치 실행 (CSV 입력 -> 임베딩 -> CSV 저장)
# -----------------------------
if __name__ == "__main__":
    import pandas as pd

    # 입력/출력 경로
    input_csv = "output/multiple/multiple_notices.csv"    # 실제 파일명/경로로 변경 가능
    config_path = "meta/notice_text_config.json"
    table_name = "notice"
    out_dir = "output/preprocessed"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if not Path(input_csv).exists():
        raise FileNotFoundError(f"입력 CSV가 없습니다: {input_csv}")

    df = pd.read_csv(input_csv)
    cfg = load_config(config_path)

    # 사용할 텍스트 컬럼
    text_cols = [c for c, v in cfg.items() if v.get("use", True)]
    present_cols = [c for c in text_cols if c in df.columns]
    if not present_cols:
        raise ValueError(f"설정된 텍스트 컬럼이 데이터에 없습니다. cfg keys={list(cfg.keys())}, df cols={list(df.columns)[:10]}...")

    # PK + 텍스트만 서브셋
    pk_cols = [c for c in TABLE_PK_MAP.get(table_name, []) if c in df.columns]
    use_cols = pk_cols + [c for c in present_cols if c not in pk_cols]
    work = df[use_cols].copy()

    # 실행
    pre = TextPreprocessor(cfg, table_pk_map=TABLE_PK_MAP)
    pre.fit(work, table_name=table_name)
    out = pre.transform(work, table_name=table_name).rename(columns=str)


    def _order_key(prefix, label):
        s = str(label)
        tail = s[len(prefix):]
        try:
            return int(tail)
        except Exception:
            return s


    for col in present_cols:
        prefix = f"{col}_emb"
        emb_cols = [c for c in out.columns if str(c).startswith(prefix)]
        if not emb_cols:
            continue
        emb_cols = sorted(emb_cols, key=lambda c: _order_key(prefix, c))
        arr = out[emb_cols].to_numpy(dtype="float32")
        out[f"{col}_embedding"] = pd.Series(
            [json.dumps(vec.tolist(), ensure_ascii=False) for vec in arr],
            index=out.index,
            dtype="string"
        )
        out.drop(columns=emb_cols, inplace=True)
    # -------------------------------------------------------------------------

    # 최종 저장
    out_csv = Path(out_dir) / f"notice_text_test.csv"
    out.to_csv(out_csv, index=False)
    print(f"✅ 저장 완료: {out_csv} (rows={len(out)}, cols={len(out.columns)})")