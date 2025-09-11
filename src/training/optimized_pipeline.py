"""
Optimized Training Pipeline with AsyncBatchPreprocessor
기존 TrainPipelineWrapper를 완전히 대체하는 고성능 파이프라인
"""
import torch
import threading
import time
import logging
from typing import Dict, Optional, Iterator
from torch.utils.data import DataLoader

from src.training.async_batch_preprocessor import AsyncBatchPreprocessor, PreprocessedBatch


class OptimizedTrainPipeline:
    """
    AsyncBatchPreprocessor를 활용한 최적화된 학습 파이프라인
    
    완전한 3단계 파이프라인:
    1. DataLoader → Raw batch (인덱스만)
    2. AsyncBatchPreprocessor → GPU-ready batch (KJT 포함)  
    3. GPU → 즉시 연산 (non_blocking 전송)
    
    기존 TrainPipelineWrapper를 완전히 대체
    """
    
    def __init__(
        self,
        dataloader: DataLoader,
        device: torch.device,
        prefetch_batches: int = 3,
        num_workers: int = 2,
        pin_memory: bool = True
    ):
        """
        Args:
            dataloader: lightweight collate_fn을 사용하는 DataLoader
            device: GPU 디바이스
            prefetch_batches: 미리 준비할 배치 개수
            num_workers: 전처리 워커 스레드 개수
            pin_memory: Pin memory 사용 여부
        """
        self.dataloader = dataloader
        self.device = device
        self.prefetch_batches = prefetch_batches
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # AsyncBatchPreprocessor 초기화
        self.preprocessor = AsyncBatchPreprocessor(
            dataset=dataloader.dataset,
            prefetch_size=prefetch_batches,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        # DataLoader 피드 스레드
        self.feeder_thread = None
        self.stop_event = threading.Event()
        
        # 성능 모니터링
        self.batch_count = 0
        self.total_batches = len(dataloader) if hasattr(dataloader, '__len__') else 0
        self.gpu_transfer_times = []
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
    def start(self):
        """파이프라인 시작"""
        if self.feeder_thread is not None:
            self.logger.warning("Pipeline already running")
            return
            
        self.logger.info("Starting optimized training pipeline...")
        
        # AsyncBatchPreprocessor 시작
        self.preprocessor.start()
        
        # DataLoader 피드 스레드 시작
        self.stop_event.clear()
        self.feeder_thread = threading.Thread(
            target=self._dataloader_feeder,
            daemon=True,
            name="DataLoader-Feeder"
        )
        self.feeder_thread.start()
        
        self.logger.info(f"Pipeline started: {self.num_workers} workers, {self.prefetch_batches} prefetch batches")
        
    def stop(self):
        """파이프라인 중지"""
        if self.feeder_thread is None:
            return
            
        self.logger.info("Stopping optimized training pipeline...")
        
        # 중지 신호
        self.stop_event.set()
        
        # 피드 스레드 대기
        if self.feeder_thread:
            self.feeder_thread.join(timeout=3.0)
            self.feeder_thread = None
            
        # AsyncBatchPreprocessor 중지
        self.preprocessor.stop()
        
        self.logger.info("Pipeline stopped successfully")
        
    def _dataloader_feeder(self):
        """DataLoader에서 raw batch를 읽어서 preprocessor에 공급"""
        self.logger.info("DataLoader feeder started")
        
        try:
            for raw_batch in self.dataloader:
                if self.stop_event.is_set():
                    break
                    
                # AsyncBatchPreprocessor에 배치 제출
                success = self.preprocessor.submit_batch(raw_batch, timeout=2.0)
                
                if not success:
                    self.logger.warning("Failed to submit batch to preprocessor")
                    # 백프레셔: 잠시 대기 후 재시도
                    time.sleep(0.1)
                    continue
                    
        except Exception as e:
            self.logger.error(f"DataLoader feeder error: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
        finally:
            self.logger.info("DataLoader feeder finished")
    
    def get_next_batch(self, timeout: float = 30.0) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        """
        GPU-ready 배치 가져오기 (즉시 사용 가능)
        
        Args:
            timeout: 최대 대기 시간
            
        Returns:
            GPU로 전송된 배치 또는 None
        """
        # 전처리 완료된 배치 가져오기
        preprocessed_batch = self.preprocessor.get_ready_batch(timeout=timeout)
        
        if preprocessed_batch is None:
            return None
            
        # GPU로 즉시 전송
        start_time = time.time()
        gpu_batch = self._transfer_to_gpu(preprocessed_batch)
        transfer_time = time.time() - start_time
        
        # 성능 추적
        self.gpu_transfer_times.append(transfer_time)
        self.batch_count += 1
        
        return gpu_batch
    
    def _transfer_to_gpu(self, preprocessed_batch: PreprocessedBatch) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        전처리된 배치를 GPU로 즉시 전송
        
        Args:
            preprocessed_batch: CPU에서 준비된 배치 (pin_memory 적용됨)
            
        Returns:
            GPU 배치
        """
        gpu_batch = {
            "notice": self._transfer_tower_data(preprocessed_batch.notice_data),
            "company": self._transfer_tower_data(preprocessed_batch.company_data)
        }
        
        return gpu_batch
    
    def _transfer_tower_data(self, tower_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """단일 타워 데이터를 GPU로 전송"""
        gpu_tower_data = {}
        
        for key, tensor in tower_data.items():
            if isinstance(tensor, torch.Tensor):
                # pin_memory가 적용된 텐서는 non_blocking=True로 빠른 전송
                gpu_tower_data[key] = tensor.to(self.device, non_blocking=self.pin_memory)
            else:
                # KJT의 경우 to() 메서드 사용
                gpu_tower_data[key] = tensor.to(self.device)
                
        return gpu_tower_data
    
    def __len__(self) -> int:
        """총 배치 개수 반환"""
        return self.total_batches
    
    def __iter__(self) -> Iterator[Dict[str, Dict[str, torch.Tensor]]]:
        """Iterator 인터페이스 제공"""
        for _ in range(self.total_batches):
            batch = self.get_next_batch()
            if batch is None:
                break
            yield batch
    
    def get_performance_stats(self) -> Dict[str, float]:
        """성능 통계 반환"""
        stats = {
            "processed_batches": self.batch_count,
            "total_batches": self.total_batches,
            "progress": self.batch_count / self.total_batches if self.total_batches > 0 else 0.0
        }
        
        # GPU 전송 시간 통계
        if self.gpu_transfer_times:
            stats.update({
                "avg_gpu_transfer_time": sum(self.gpu_transfer_times) / len(self.gpu_transfer_times),
                "min_gpu_transfer_time": min(self.gpu_transfer_times),
                "max_gpu_transfer_time": max(self.gpu_transfer_times)
            })
        
        # AsyncBatchPreprocessor 통계
        preprocessor_stats = self.preprocessor.get_stats()
        stats.update({f"preprocessor_{k}": v for k, v in preprocessor_stats.items()})
        
        return stats
    
    def __del__(self):
        """소멸자에서 리소스 정리"""
        self.stop()


# 기존 TrainPipelineWrapper와의 호환성을 위한 별칭
TrainPipelineWrapper = OptimizedTrainPipeline