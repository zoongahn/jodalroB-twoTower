#!/usr/bin/env python3
"""
기존 predict.py 출력 JSON 파일을 새로운 형식으로 변환

기존 형식:
{
  "notice_id": ["20210408676", "000"],
  "notice_title": "공립행복어린이집 그린리모델링(건축)",
  "top_k_companies": [["6168618105", 0.0864897295832634], ...],
  "company_names": {"6168618105": "주식회사 초이스건설", ...}
}

새 형식:
{
  "notice": {
    "bidntceno": "20210408676",
    "bidntceord": "000",
    "title": "공립행복어린이집 그린리모델링(건축)",
    "embedding": [...]  # 옵션
  },
  "top_k_companies": {
    "6168618105": {
      "similarity": 0.0864897295832634,
      "name": "주식회사 초이스건설",
      "embedding": [...]  # 옵션
    },
    ...
  }
}
"""

import json
import sys
from pathlib import Path


def convert_json_format(input_path: str, output_path: str = None):
    """
    기존 JSON 형식을 새로운 형식으로 변환

    Args:
        input_path: 입력 JSON 파일 경로
        output_path: 출력 JSON 파일 경로 (None이면 입력 파일 덮어쓰기)
    """
    # JSON 파일 로드
    with open(input_path, 'r', encoding='utf-8') as f:
        old_format = json.load(f)

    # notice_id 추출
    notice_id = old_format.get('notice_id', [])
    if isinstance(notice_id, list):
        bidntceno, bidntceord = notice_id[0], notice_id[1] if len(notice_id) > 1 else "000"
    else:
        # notice_id가 문자열이면 분리 시도
        parts = str(notice_id).split('_')
        bidntceno, bidntceord = parts[0], parts[1] if len(parts) > 1 else "000"

    # Notice 정보 구성
    notice_info = {
        "bidntceno": bidntceno,
        "bidntceord": bidntceord,
        "title": old_format.get('notice_title')
    }

    # Notice 임베딩이 있으면 추가
    if 'notice_embedding' in old_format:
        notice_info['embedding'] = old_format['notice_embedding']

    # Company 정보 구성
    top_k_companies_dict = {}
    company_names = old_format.get('company_names', {})
    company_embeddings = old_format.get('company_embeddings', {})

    for company_id, similarity_score in old_format.get('top_k_companies', []):
        company_info = {
            "similarity": similarity_score,
            "name": company_names.get(company_id, None)
        }

        # Company 임베딩이 있으면 추가
        if company_id in company_embeddings:
            company_info['embedding'] = company_embeddings[company_id]

        top_k_companies_dict[company_id] = company_info

    # 최종 JSON 구조
    new_format = {
        "notice": notice_info,
        "top_k_companies": top_k_companies_dict
    }

    # 저장
    if output_path is None:
        output_path = input_path

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_format, f, ensure_ascii=False, indent=2)

    print(f"✓ 변환 완료: {output_path}")
    print(f"  - Notice: {bidntceno}-{bidntceord}")
    print(f"  - Companies: {len(top_k_companies_dict)}개")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python convert_json_format.py <input_json> [output_json]")
        print("예제: python convert_json_format.py output.json")
        print("     python convert_json_format.py input.json output_new.json")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    convert_json_format(input_file, output_file)
