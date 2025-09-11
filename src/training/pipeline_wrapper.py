import torch
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Any, List
from torch.utils.data import DataLoader
import time
import logging

# 기존 KJT 함수들 import
from src.towers.pairs.unified_bid_data_loader import _build_batch_kjt, _build_batch_kjt_gpu


class TrainPipelineWrapper:
    """
    GPU 점유율 극대화를 위한 학습 파이프라인 래퍼
    
    백그라운드에서 데이터 전처리를 수행하여 GPU가 대기 없이 연속 연산을 수행할 수 있도록 함.
    CPU 전처리(KJT 생성, 텐서 변환 등)와 GPU 연산을 완전히 분리하여 파이프라인화.
    """
    
    def __init__(
        self,
        dataloader: DataLoader,
        device: torch.device,
        prefetch_batches: int = 3,
        use_gpu_kjt: bool = False,
        pin_memory: bool = True
    ):
        """
        Args:
            dataloader: 원본 데이터로더
            device: GPU 디바이스
            prefetch_batches: 미리 준비할 배치 개수
            use_gpu_kjt: GPU에서 KJT 생성 여부
            pin_memory: Pin memory 사용 여부
        """
        self.dataloader = dataloader
        self.device = device
        self.prefetch_batches = prefetch_batches
        self.use_gpu_kjt = use_gpu_kjt
        self.pin_memory = pin_memory
        
        # 큐 설정 - 메모리 오버플로우 방지
        self.prepared_queue = queue.Queue(maxsize=prefetch_batches)
        self.error_queue = queue.Queue()
        
        # 백그라운드 스레드 관리
        self.background_thread = None
        self.stop_event = threading.Event()
        
        # 재사용 가능한 ThreadPool (오버헤드 제거)
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
        
        # 성능 모니터링
        self.batch_count = 0
        self.prep_times = []
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def start_background_prep(self):
        """백그라운드 데이터 준비 시작"""
        if self.background_thread is not None:
            self.logger.warning("Background thread already running")
            return
            
        self.stop_event.clear()
        self.background_thread = threading.Thread(
            target=self._background_data_preparation,
            daemon=True
        )
        self.background_thread.start()
        self.logger.info("Background data preparation started")
        
    def stop_background_prep(self):
        """백그라운드 데이터 준비 중지"""
        if self.background_thread is None:
            return
            
        self.stop_event.set()
        
        # 큐에 None을 넣어서 대기 중인 get() 해제
        try:
            self.prepared_queue.put(None, timeout=1.0)
        except queue.Full:
            pass
            
        self.background_thread.join(timeout=2.0)
        self.background_thread = None
        
        # ThreadPool 정리
        self.thread_pool.shutdown(wait=True)
        
        self.logger.info("Background data preparation stopped")
        
    def _background_data_preparation(self):
        """백그라운드에서 배치 전처리 수행"""
        try:
            for batch in self.dataloader:
                if self.stop_event.is_set():
                    break
                    
                start_time = time.time()
                
                # DataLoader는 이미 collate된 배치를 반환하므로 그대로 사용
                prepared_batch = batch
                
                prep_time = time.time() - start_time
                self.prep_times.append(prep_time)
                
                # 준비된 배치를 큐에 추가 (블로킹 가능)
                if not self.stop_event.is_set():
                    self.prepared_queue.put(prepared_batch)
                    
        except Exception as e:
            self.error_queue.put(e)
            self.logger.error(f"Background preparation error: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
        
    def get_next_batch(self) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        """GPU용 준비된 배치 가져오기 (즉시 사용 가능)"""
        try:
            # 에러 체크
            if not self.error_queue.empty():
                error = self.error_queue.get()
                raise error
                
            # 준비된 배치 가져오기 (타임아웃으로 무한 대기 방지)
            batch = self.prepared_queue.get(timeout=30.0)
            
            if batch is None:  # 종료 신호
                return None
                
            # GPU로 전송 (pin_memory 사용 시 더 빠름)
            gpu_batch = self._move_to_gpu(batch)
            
            self.batch_count += 1
            return gpu_batch
            
        except queue.Empty:
            self.logger.warning("Batch preparation timeout - GPU may be faster than CPU preprocessing")
            return None
            
    def _move_to_gpu(self, batch: Dict) -> Dict:
        """배치를 GPU로 이동 (pin_memory 사용 시 더 빠름)"""
        gpu_batch = {}
        
        for key, data in batch.items():
            if isinstance(data, dict):
                gpu_batch[key] = {}
                for sub_key, tensor in data.items():
                    if isinstance(tensor, torch.Tensor):
                        # pin_memory 사용 시 non_blocking=True로 더 빠른 전송
                        gpu_batch[key][sub_key] = tensor.to(self.device, non_blocking=self.pin_memory)
                    else:
                        # KJT의 경우 to() 메서드 사용
                        gpu_batch[key][sub_key] = tensor.to(self.device)
            else:
                # 텐서가 직접 있는 경우
                if isinstance(data, torch.Tensor):
                    gpu_batch[key] = data.to(self.device, non_blocking=self.pin_memory)
                else:
                    gpu_batch[key] = data.to(self.device)
                    
        return gpu_batch
        
    def __len__(self):
        """DataLoader 길이 반환"""
        return len(self.dataloader)
        
    def get_performance_stats(self) -> Dict[str, float]:
        """성능 통계 반환"""
        if not self.prep_times:
            return {}
            
        return {
            "avg_prep_time": sum(self.prep_times) / len(self.prep_times),
            "min_prep_time": min(self.prep_times),
            "max_prep_time": max(self.prep_times),
            "total_batches": self.batch_count,
            "queue_size": self.prepared_queue.qsize()
        }
        
    def __del__(self):
        """소멸자에서 리소스 정리"""
        self.stop_background_prep()