import torch
from src.towers.two_tower_model import create_two_tower_model
from src.torchrec_preprocess.schema import build_torchrec_schema_from_meta
from src.torchrec_preprocess.torchrec_inputs import get_tower_input
from data.database_connector import DatabaseConnector


def test_two_tower_model():
    """TwoTowerModel 간단 테스트"""
    print("TwoTowerModel 테스트 시작...")
    
    # DB 연결 및 스키마 구축
    db = DatabaseConnector()
    engine = db.engine
    
    config = {
        "notice_table": "notice",
        "company_table": "company", 
        "pair_table": "bid_two_tower",
        "pair_notice_id_cols": ["bidntceno", "bidntceord"],
        "pair_company_id_cols": ["bizno"],
        "metadata_path": "meta/metadata.csv"
    }
    schema = build_torchrec_schema_from_meta(**config)
    
    # 입력 데이터 생성
    print("입력 데이터 준비...")
    notice_input = get_tower_input(
        db_engine=engine,
        schema=schema.notice, 
        tower_name="Notice",
        chunksize=1000,
        limit=10000,  # 빠른 테스트를 위해 작게
        show_progress=True
    )
    
    company_input = get_tower_input(
        db_engine=engine,
        schema=schema.company,
        tower_name="Company", 
        chunksize=1000,
        limit=10000,
        show_progress=True
    )
    
    print(f"Notice 배치 크기: {notice_input['dense'].shape[0]}")
    print(f"Company 배치 크기: {company_input['dense'].shape[0]}")
    
    # TwoTowerModel 생성
    print("TwoTowerModel 생성...")
    two_tower = create_two_tower_model(
        notice_categorical_keys=schema.notice.categorical,
        company_categorical_keys=schema.company.categorical,
        metadata_path=config["metadata_path"],
        categorical_embedding_dim=64,
        notice_dense_input_dim=256,
        company_dense_input_dim=128,
        tower_hidden_dims=[256, 128],
        final_embedding_dim=128,
        dropout_rate=0.2,
        device=torch.device("cuda:0")
    )
    
    print(f"총 파라미터: {sum(p.numel() for p in two_tower.parameters()):,}")
    
    # Forward Pass 테스트
    print("\n[Forward Pass 테스트]")
    try:
        # 기본 forward (임베딩만 반환)
        notice_emb, company_emb = two_tower(notice_input, company_input)
        print(f"Notice 임베딩: {notice_emb.shape}")
        print(f"Company 임베딩: {company_emb.shape}")
        print(f"L2 norm 확인: Notice={torch.norm(notice_emb[0]):.4f}, Company={torch.norm(company_emb[0]):.4f}")
        
        # 유사도 포함 forward
        result = two_tower(notice_input, company_input, return_similarity=True)
        similarity_matrix = result["similarity_matrix"]
        print(f"유사도 행렬: {similarity_matrix.shape}")
        print(f"대각선 평균: {torch.diag(similarity_matrix).mean():.4f}")
        print(f"전체 평균: {similarity_matrix.mean():.4f}")
        
        print("Forward pass 성공!")
        
    except Exception as e:
        print(f"Forward pass 실패: {e}")
        return
    
    # Gradient 테스트
    print("\n[Gradient 테스트]")
    try:
        two_tower.train()
        
        # 더미 손실 계산
        notice_emb, company_emb = two_tower(notice_input, company_input)
        similarity_matrix = torch.mm(notice_emb, company_emb.t())
        
        # 대각선을 positive로 하는 cross entropy loss
        labels = torch.arange(len(notice_emb)).to(torch.device("cuda:0"))
        loss = torch.nn.functional.cross_entropy(similarity_matrix, labels)
        
        print(f"더미 손실: {loss.item():.4f}")
        
        # Backward
        loss.backward()
        
        # Gradient 확인
        grad_count = 0
        grad_norms = []
        for name, param in two_tower.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                grad_count += 1
        
        print(f"총 {grad_count}개 파라미터에 gradient 계산됨")
        print(f"평균 gradient norm: {sum(grad_norms)/len(grad_norms):.6f}")
        print("Gradient 테스트 성공!")
        
    except Exception as e:
        print(f"Gradient 테스트 실패: {e}")
    
    # Top-K 예측 테스트
    print("\n[Top-K 예측 테스트]")
    try:
        k = 5
        top_similarities, top_indices = two_tower.predict_top_k(
            notice_input, company_input, k=k
        )
        print(f"Top-{k} 유사도: {top_similarities.shape}")
        print(f"Top-{k} 인덱스: {top_indices.shape}")
        print(f"첫 번째 공고의 Top-{k} 유사도: {top_similarities[0]}")
        print("Top-K 예측 성공!")
        
    except Exception as e:
        print(f"Top-K 예측 실패: {e}")
    
    print("\nTwoTowerModel 테스트 완료!")


if __name__ == "__main__":
    test_two_tower_model()