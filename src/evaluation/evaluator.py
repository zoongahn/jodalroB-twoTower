import torch
from typing import Dict, List, Any
from tqdm import tqdm


class TwoTowerEvaluator:
    """
    Two-Tower 모델 평가를 위한 클래스
    - Recall@K, MRR, Accuracy 등 다양한 지표 계산
    - 단일 배치 및 전체 데이터 평가 지원
    """
    
    def __init__(self, device: str = "cuda:0"):
        """
        Args:
            device: 계산에 사용할 디바이스
        """
        self.device = device
        
    def compute_recall_at_k(self, similarity_matrix: torch.Tensor, k: int) -> torch.Tensor:
        """
        Top-K Recall 계산
        
        Args:
            similarity_matrix: 유사도 행렬 [B, B]
            k: Top-K의 K값
            
        Returns:
            recall@k 점수
        """
        batch_size = similarity_matrix.size(0)
        
        # Top-K 인덱스 추출
        actual_k = min(k, similarity_matrix.size(1))
        top_k_indices = torch.topk(similarity_matrix, k=actual_k, dim=1).indices  # [B, actual_k]
        
        # 정답 인덱스 (대각선)
        correct_indices = torch.arange(batch_size, device=similarity_matrix.device).unsqueeze(1)  # [B, 1]
        
        # Top-K에 정답이 포함되었는지 확인
        recall = (top_k_indices == correct_indices).any(dim=1).float().mean()
        
        return recall

    def compute_mrr(self, similarity_matrix: torch.Tensor) -> torch.Tensor:
        """
        Mean Reciprocal Rank 계산
        
        Args:
            similarity_matrix: 유사도 행렬 [B, B]
            
        Returns:
            MRR 점수
        """
        batch_size = similarity_matrix.size(0)
        
        # 각 행을 내림차순으로 정렬한 인덱스
        sorted_indices = torch.argsort(similarity_matrix, dim=1, descending=True)  # [B, B]
        
        # 각 정답의 순위 찾기
        correct_indices = torch.arange(batch_size, device=similarity_matrix.device)
        
        reciprocal_ranks = []
        for i in range(batch_size):
            # i번째 공고의 정답 업체(i)가 몇 번째 순위인지 찾기
            rank_position = (sorted_indices[i] == correct_indices[i]).nonzero().item()
            reciprocal_rank = 1.0 / (rank_position + 1)  # 순위는 1부터 시작
            reciprocal_ranks.append(reciprocal_rank)
        
        mrr = torch.tensor(reciprocal_ranks, device=similarity_matrix.device).mean()
        return mrr

    def compute_comprehensive_metrics(self, similarity_matrix: torch.Tensor, basic_metrics: Dict) -> Dict[str, float]:
        """
        종합적인 평가 지표 계산
        
        Args:
            similarity_matrix: 유사도 행렬
            basic_metrics: 기본 메트릭 (loss, accuracy 등)
            
        Returns:
            모든 평가 지표가 담긴 딕셔너리
        """
        batch_size = similarity_matrix.size(0)
        
        # 추가 지표 계산
        recall_5 = self.compute_recall_at_k(similarity_matrix, k=5).item()
        recall_10 = self.compute_recall_at_k(similarity_matrix, k=10).item()
        mrr = self.compute_mrr(similarity_matrix).item()
        
        # 랜덤 기준선
        random_accuracy = 1.0 / batch_size
        random_recall_5 = min(5.0 / batch_size, 1.0)
        random_recall_10 = min(10.0 / batch_size, 1.0)
        
        metrics = {
            # 기본 지표
            'loss': basic_metrics.get('loss', 0.0),
            'accuracy': basic_metrics.get('accuracy', 0.0),
            'similarity_gap': basic_metrics.get('similarity_gap', 0.0),
            'positive_similarity_mean': basic_metrics.get('positive_similarity_mean', 0.0),
            'negative_similarity_mean': basic_metrics.get('negative_similarity_mean', 0.0),
            
            # 추가 지표
            'recall@5': recall_5,
            'recall@10': recall_10,
            'mrr': mrr,
            'batch_size': batch_size,
            
            # 랜덤 기준선
            'random_accuracy': random_accuracy,
            'random_recall@5': random_recall_5,
            'random_recall@10': random_recall_10,
            
            # 개선도
            'accuracy_improvement': basic_metrics.get('accuracy', 0.0) > random_accuracy,
            'recall@5_improvement': recall_5 > random_recall_5,
            'recall@10_improvement': recall_10 > random_recall_10,
        }
        
        return metrics

    def evaluate_single_batch(self, model, batch: Dict, verbose: bool = True) -> Dict[str, float]:
        """
        단일 배치에 대한 종합 평가
        
        Args:
            model: 평가할 모델
            batch: 입력 배치
            verbose: 결과 출력 여부
            
        Returns:
            평가 지표 딕셔너리
        """
        model.eval()
        with torch.no_grad():
            result = model(batch, return_metrics=True)
            similarity_matrix = result['similarity_matrix']
            
            # 기본 메트릭 추출
            basic_metrics = {
                'loss': result['loss'].item(),
                'accuracy': result['accuracy'].item(),
                'similarity_gap': result['similarity_gap'].item(),
                'positive_similarity_mean': result['positive_similarity_mean'].item(),
                'negative_similarity_mean': result['negative_similarity_mean'].item(),
            }
            
            # 종합 지표 계산
            metrics = self.compute_comprehensive_metrics(similarity_matrix, basic_metrics)
            
            if verbose:
                self.print_single_batch_results(metrics)
                
            return metrics

    def evaluate_comprehensive(self, model, dataloader, verbose: bool = True) -> Dict[str, float]:
        """
        전체 데이터에 대한 종합 평가
        
        Args:
            model: 평가할 모델
            dataloader: 데이터로더
            verbose: 결과 출력 여부
            
        Returns:
            평균 평가 지표 딕셔너리
        """
        model.eval()
        
        all_metrics = {
            'losses': [],
            'accuracies': [],
            'recall@5': [],
            'recall@10': [],
            'mrrs': [],
            'similarity_gaps': []
        }
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Comprehensive Evaluation") if verbose else dataloader
            
            for batch in pbar:
                result = model(batch, return_metrics=True)
                similarity_matrix = result['similarity_matrix']
                
                # 지표 수집
                all_metrics['losses'].append(result['loss'].item())
                all_metrics['accuracies'].append(result['accuracy'].item())
                all_metrics['recall@5'].append(self.compute_recall_at_k(similarity_matrix, k=5).item())
                all_metrics['recall@10'].append(self.compute_recall_at_k(similarity_matrix, k=10).item())
                all_metrics['mrrs'].append(self.compute_mrr(similarity_matrix).item())
                all_metrics['similarity_gaps'].append(result['similarity_gap'].item())
        
        # 평균 계산
        avg_metrics = {
            'loss': sum(all_metrics['losses']) / len(all_metrics['losses']),
            'accuracy': sum(all_metrics['accuracies']) / len(all_metrics['accuracies']),
            'recall@5': sum(all_metrics['recall@5']) / len(all_metrics['recall@5']),
            'recall@10': sum(all_metrics['recall@10']) / len(all_metrics['recall@10']),
            'mrr': sum(all_metrics['mrrs']) / len(all_metrics['mrrs']),
            'similarity_gap': sum(all_metrics['similarity_gaps']) / len(all_metrics['similarity_gaps']),
            'num_batches': len(dataloader),
        }
        
        if verbose:
            self.print_comprehensive_results(avg_metrics)
            
        return avg_metrics

    def print_single_batch_results(self, metrics: Dict[str, float]):
        """단일 배치 평가 결과 출력"""
        print(f"배치 크기: {int(metrics['batch_size'])}")
        print(f"Loss: {metrics['loss']:.4f}")
        print(f"Top-1 Accuracy: {metrics['accuracy']:.3f}")
        print(f"Recall@5: {metrics['recall@5']:.3f}")
        print(f"Recall@10: {metrics['recall@10']:.3f}")
        print(f"MRR: {metrics['mrr']:.3f}")
        print(f"Similarity Gap: {metrics['similarity_gap']:.3f}")
        print(f"Positive Similarity (평균): {metrics['positive_similarity_mean']:.3f}")
        print(f"Negative Similarity (평균): {metrics['negative_similarity_mean']:.3f}")
        
        print(f"\n--- 랜덤 기준선과 비교 ---")
        print(f"랜덤 Top-1 정확도: {metrics['random_accuracy']:.3f} | 현재: {metrics['accuracy']:.3f} ({'개선' if metrics['accuracy_improvement'] else '미흡'})")
        print(f"랜덤 Recall@5: {metrics['random_recall@5']:.3f} | 현재: {metrics['recall@5']:.3f} ({'개선' if metrics['recall@5_improvement'] else '미흡'})")
        print(f"랜덤 Recall@10: {metrics['random_recall@10']:.3f} | 현재: {metrics['recall@10']:.3f} ({'개선' if metrics['recall@10_improvement'] else '미흡'})")

    def print_comprehensive_results(self, metrics: Dict[str, float]):
        """종합 평가 결과 출력"""
        print(f"테스트 배치 수: {int(metrics['num_batches'])}")
        print(f"평균 Loss: {metrics['loss']:.4f}")
        print(f"평균 Top-1 Accuracy: {metrics['accuracy']:.3f}")
        print(f"평균 Recall@5: {metrics['recall@5']:.3f}")
        print(f"평균 Recall@10: {metrics['recall@10']:.3f}")
        print(f"평균 MRR: {metrics['mrr']:.3f}")
        print(f"평균 Similarity Gap: {metrics['similarity_gap']:.3f}")
        
        print(f"\n--- 성능 평가 ---")
        self.print_performance_assessment(metrics)

    def print_performance_assessment(self, metrics: Dict[str, float]):
        """성능 평가 결과 출력"""
        accuracy = metrics['accuracy']
        recall_10 = metrics['recall@10']
        similarity_gap = metrics['similarity_gap']
        
        # Top-1 정확도 평가
        if accuracy > 0.3:
            print("Top-1 정확도: 우수 (0.3 이상)")
        elif accuracy > 0.15:
            print("Top-1 정확도: 보통 (0.15~0.3)")
        else:
            print("Top-1 정확도: 미흡 (0.15 미만)")
        
        # Recall@10 평가
        if recall_10 > 0.6:
            print("Recall@10: 실용적 수준 (0.6 이상)")
        elif recall_10 > 0.4:
            print("Recall@10: 개선 필요 (0.4~0.6)")
        else:
            print("Recall@10: 부족 (0.4 미만)")
            
        # 유사도 구분 평가
        if similarity_gap > 0.5:
            print("유사도 구분: 양호 (0.5 이상)")
        else:
            print("유사도 구분: 개선 필요 (0.5 미만)")

    def demonstrate_predictions(self, model, batch: Dict, top_k: int = 10):
        """예측 결과 시연"""
        model.eval()
        with torch.no_grad():
            predictions = model.predict_batch(batch, top_k=top_k)
            
            print(f"--- 추론 예제 ---")
            print(f"배치 크기: {batch['notice']['dense'].shape[0]}")
            print(f"첫 번째 공고의 Top-5 유사도: {predictions['top_similarities'][0][:5]}")
            print(f"첫 번째 공고의 Top-5 업체 인덱스: {predictions['top_indices'][0][:5]}")
            
            # 유사도 행렬 분석
            similarity_matrix = predictions['all_similarities']
            print(f"유사도 행렬 크기: {similarity_matrix.shape}")
            print(f"대각선 유사도 (positive pairs): {torch.diag(similarity_matrix)[:5]}")
            print(f"첫 번째 행 유사도 범위: {similarity_matrix[0].min():.3f} ~ {similarity_matrix[0].max():.3f}")