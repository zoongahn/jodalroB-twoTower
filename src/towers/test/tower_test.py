import torch
import torch.nn as nn
from sqlalchemy import Engine
from typing import Dict, Any

# 필요한 모듈들 import
from src.torchrec_preprocess.schema import build_torchrec_schema_from_meta
from src.torchrec_preprocess.torchrec_inputs import get_tower_input
from data.database_connector import DatabaseConnector

# 타워 모듈들 import
from src.towers.tower.base_tower import BaseTower
from src.towers.tower.notice_tower import NoticeTower
from src.towers.tower.company_tower import CompanyTower



def test_single_tower(
    tower_name: str,
    tower_model: BaseTower, 
    tower_input: Dict[str, Any],
    test_forward: bool = True,
    test_gradient: bool = True
):
    """단일 타워 테스트"""
    print(f"\n{'='*60}")
    print(f"Testing {tower_name} Tower")
    print(f"{'='*60}")
    
    # 1. 모델 기본 정보
    total_params = sum(p.numel() for p in tower_model.parameters())
    trainable_params = sum(p.numel() for p in tower_model.parameters() if p.requires_grad)
    
    print(f"총 파라미터: {total_params:,}")
    print(f"학습 가능한 파라미터: {trainable_params:,}")
    print(f"모델 디바이스: {next(tower_model.parameters()).device}")
    
    # 2. 입력 데이터 확인
    print(f"\n[입력 데이터 확인]")
    if tower_input.get('dense') is not None:
        dense = tower_input['dense']
        print(f"Dense: {dense.shape} | 디바이스: {dense.device}")
        print(f"Dense 범위: min={dense.min():.4f}, max={dense.max():.4f}")
    
    if tower_input.get('kjt') is not None:
        kjt = tower_input['kjt']
        print(f"KJT: {len(kjt.keys())} keys, values={kjt.values().shape}")
        print(f"KJT 디바이스: {kjt.device()}")
    
    if not test_forward:
        return
    
# 3. Forward 테스트 - torch.no_grad() 제거!
    print(f"\n[Forward Pass 테스트]")
    try:
        # no_grad() 제거하여 gradient 추적 활성화
        output = tower_model(tower_input)
        
        print(f"출력 shape: {output.shape}")
        print(f"출력 디바이스: {output.device}")
        print(f"출력 requires_grad: {output.requires_grad}")  # 추가
        print(f"출력 범위: min={output.min():.4f}, max={output.max():.4f}")
        print(f"L2 norm (첫 3개 샘플): {torch.norm(output[:3], dim=1)}")
        print("Forward pass 성공!")
        
    except Exception as e:
        print(f"Forward pass 실패: {e}")
        return
    
    if not test_gradient:
        return
    
    # 4. Gradient 테스트 - 개선된 버전
    print(f"\n[Gradient 테스트]")
    try:
        tower_model.train()  # 훈련 모드로 전환
        
        # 새로운 forward pass (gradient 추적 상태에서)
        output = tower_model(tower_input)
        print(f"Gradient 테스트용 출력 requires_grad: {output.requires_grad}")
        
        # 더미 타겟 생성
        target_embeddings = torch.randn_like(output)
        
        # 코사인 유사도 기반 손실 (더 안정적)
        output_normalized = torch.nn.functional.normalize(output, p=2, dim=1)
        target_normalized = torch.nn.functional.normalize(target_embeddings, p=2, dim=1)
        similarities = torch.sum(output_normalized * target_normalized, dim=1)
        loss = -similarities.mean()  # 유사도를 최대화하는 손실
        
        print(f"더미 손실: {loss.item():.4f}")
        print(f"손실 requires_grad: {loss.requires_grad}")
        
        # Backward
        loss.backward()
        
        # Gradient 확인
        grad_norms = []
        grad_count = 0
        for name, param in tower_model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                grad_count += 1
                if grad_count <= 5:  # 처음 5개만 출력
                    print(f"{name}: grad_norm={grad_norm:.6f}")
        
        if grad_norms:
            print(f"총 {grad_count}개 파라미터에 gradient 계산됨")
            print(f"평균 gradient norm: {sum(grad_norms)/len(grad_norms):.6f}")
            print("Gradient 계산 성공!")
        else:
            print("❌ Gradient가 계산되지 않았습니다.")
            # 추가 디버깅
            print("모델 파라미터 상태 체크:")
            for name, param in tower_model.named_parameters():
                print(f"  {name}: requires_grad={param.requires_grad}, shape={param.shape}")
            
    except Exception as e:
        print(f"Gradient 테스트 실패: {e}")
        import traceback
        traceback.print_exc()


def main():
    """메인 테스트 함수"""
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
    
    # 테스트 설정
    test_params = {
        "chunksize": 1000,
        "limit": 1000,  # 작은 데이터로 빠른 테스트
        "show_progress": True
    }
    
    # Notice Tower 테스트
    print("Notice Tower 테스트 시작...")
    notice_input = get_tower_input(
        db_engine=engine,
        schema=schema.notice, 
        tower_name="Notice",
        **test_params
    )
    
    notice_tower = NoticeTower(
        categorical_keys=schema.notice.categorical,
        metadata_path=config["metadata_path"],
        categorical_embedding_dim=64,
        dense_input_dim=256,
        tower_hidden_dims=[256, 128],
        final_embedding_dim=128,
        dropout_rate=0.2,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    
    test_single_tower("Notice", notice_tower, notice_input)
    
    # Company Tower 테스트
    print("\n\nCompany Tower 테스트 시작...")
    company_input = get_tower_input(
        db_engine=engine,
        schema=schema.company,
        tower_name="Company", 
        **test_params
    )
    
    company_tower = CompanyTower(
        categorical_keys=schema.company.categorical,
        metadata_path=config["metadata_path"],
        categorical_embedding_dim=64,  # Notice와 동일
        dense_input_dim=128,
        tower_hidden_dims=[256, 128],  # Notice와 동일하거나 비슷하게
        final_embedding_dim=128,       # 반드시 동일
        dropout_rate=0.2,              # Notice와 동일
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    
    test_single_tower("Company", company_tower, company_input)
    
    # 타워 간 임베딩 유사도 테스트
    print("\n\n" + "="*60)
    print("타워 간 임베딩 유사도 테스트")
    print("="*60)
    
    try:
        with torch.no_grad():
            notice_emb = notice_tower(notice_input)[:10]  # 처음 10개만
            company_emb = company_tower(company_input)[:10]
            
            # 코사인 유사도 계산
            similarities = torch.mm(notice_emb, company_emb.t())
            print(f"유사도 행렬 shape: {similarities.shape}")
            print(f"유사도 범위: {similarities.min():.4f} ~ {similarities.max():.4f}")
            print(f"유사도 행렬: {similarities}")
            print("타워 간 유사도 계산 성공!")
            
    except Exception as e:
        print(f"유사도 계산 실패: {e}")


if __name__ == "__main__":
    main()