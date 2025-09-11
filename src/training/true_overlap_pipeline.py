"""
True Overlap Pipeline for Real Batch-Level Parallel Processing
진짜 오버랩: GPU가 배치 N 처리 중 배치 N+1,N+2,N+3 백그라운드 준비
"""
import torch
import threading
import queue
import time
import itertools
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Iterator
from torch.utils.data import DataLoader

from src.towers.pairs.unified_bid_data_loader import _build_batch_kjt
from src.training.fast_kjt_builder import build_kjt_from_numpy


class StreamingDataLoaderManager:
    """
    DataLoader를 별도 스레드에서 논스톱 실행
    페어 로딩(청크)은 건드리지 않고, 배치만 연속 공급
    """
    
    def __init__(self, dataloader: DataLoader, buffer_size: int = 8):
        """
        Args:
            dataloader: lightweight collate_fn 사용하는 DataLoader
            buffer_size: raw 배치 버퍼 크기
        """
        self.dataloader = dataloader
        self.buffer_size = buffer_size
        
        # Raw 배치 큐 (인덱스만 있는 배치들)
        self.raw_queue = queue.Queue(maxsize=buffer_size)
        self.error_queue = queue.Queue()
        
        # 스레드 관리
        self.loader_thread = None
        self.stop_event = threading.Event()
        self.total_batches = len(dataloader) if hasattr(dataloader, '__len__') else 0
        self.loaded_count = 0
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)  # INFO 레벨로 변경
    
    def start(self):
        """연속적 배치 로딩 시작"""
        if self.loader_thread is not None:
            self.logger.warning("DataLoader already streaming")
            return
            
        # DataLoader 정보 확인
        self.logger.info(f"DataLoader info: total_batches={self.total_batches}, has_dataset={hasattr(self.dataloader, 'dataset')}")
        if hasattr(self.dataloader, 'dataset'):
            dataset = self.dataloader.dataset
            self.logger.info(f"Dataset info: type={type(dataset).__name__}, len={len(dataset) if hasattr(dataset, '__len__') else 'N/A'}")
            
        self.stop_event.clear()
        self.loader_thread = threading.Thread(
            target=self._continuous_loading,
            daemon=True,
            name="DataLoader-Stream"
        )
        self.loader_thread.start()
        self.logger.info(f"Started continuous DataLoader streaming (buffer: {self.buffer_size})")
    
    def stop(self):
        """스트리밍 중지"""
        if self.loader_thread is None:
            return
            
        self.stop_event.set()
        
        # 큐 비우기
        while not self.raw_queue.empty():
            try:
                self.raw_queue.get_nowait()
            except queue.Empty:
                break
        
        if self.loader_thread:
            self.loader_thread.join(timeout=2.0)
            self.loader_thread = None
            
        self.logger.info("DataLoader streaming stopped")
    
    def _continuous_loading(self):
        """백그라운드에서 DataLoader 연속 실행"""
        try:
            self.logger.info(f"Starting continuous loading from DataLoader (total: {self.total_batches})")
            
            # Iterator 직접 생성
            dataloader_iter = iter(self.dataloader)
            self.logger.info("DataLoader iterator created successfully")
            
            while not self.stop_event.is_set():
                try:
                    # next()로 배치 가져오기
                    raw_batch = next(dataloader_iter)
                    
                    # 디버깅: 배치 로드 확인
                    if self.loaded_count == 0:
                        self.logger.info(f"First batch loaded successfully, size: {len(raw_batch) if raw_batch else 0}")
                        
                    # Raw 배치를 큐에 추가 (블로킹 가능)
                    self.raw_queue.put(raw_batch, block=True)
                    self.loaded_count += 1
                    
                    # 진행률 로깅 (더 자주)
                    if self.loaded_count % 10 == 0:  # 100 -> 10으로 변경
                        if self.total_batches > 0:
                            progress = (self.loaded_count / self.total_batches) * 100
                            self.logger.info(f"DataLoader progress: {self.loaded_count}/{self.total_batches} ({progress:.1f}%), queue size: {self.raw_queue.qsize()}")
                            
                except StopIteration:
                    self.logger.info("DataLoader iteration completed")
                    break
                    
        except Exception as e:
            self.error_queue.put(e)
            self.logger.error(f"DataLoader streaming error: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
        finally:
            self.logger.info(f"DataLoader streaming finished: {self.loaded_count} batches loaded")
    
    def get_raw_batch(self, timeout: float = 30.0) -> Optional[List[Dict]]:
        """Raw 배치 가져오기"""
        try:
            # 에러 체크
            if not self.error_queue.empty():
                error = self.error_queue.get_nowait()
                raise error
                
            return self.raw_queue.get(timeout=timeout)
            
        except queue.Empty:
            self.logger.warning("Raw batch queue timeout")
            return None
    
    def qsize(self) -> int:
        """현재 큐 크기"""
        return self.raw_queue.qsize()


class BatchProcessorPool:
    """
    여러 배치를 동시에 병렬 처리
    핵심: 인덱싱 + KJT 생성만 담당 (피쳐 데이터는 이미 메모리에 있음)
    """
    
    def __init__(
        self, 
        dataset, 
        num_workers: int = 12,  # 4 -> 12로 대폭 증가
        ready_queue_size: int = 24,  # 12 -> 24로 증가
        pin_memory: bool = True
    ):
        """
        Args:
            dataset: 피쳐 데이터가 미리 로딩된 UnifiedBidDataset
            num_workers: 병렬 처리 워커 수
            ready_queue_size: GPU-ready 배치 큐 크기
            pin_memory: pin_memory 적용 여부
        """
        self.dataset = dataset
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # GPU-ready 배치 큐
        self.ready_queue = queue.Queue(maxsize=ready_queue_size)
        self.error_queue = queue.Queue()
        
        # 워커 풀
        self.worker_pool = ThreadPoolExecutor(max_workers=num_workers)
        self.processing_futures = {}  # future -> batch_info 매핑
        
        # 성능 추적
        self.processed_count = 0
        self.processing_times = []
        self.stats_lock = threading.Lock()
        
        # 프로파일링을 위한 세부 시간 측정
        self.indexing_times = []
        self.kjt_times = []
        self.pin_times = []
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)  # INFO 레벨로 변경
        
    def process_batch_async(self, raw_batch: List[Dict]) -> bool:
        """
        배치를 비동기로 처리 시작
        
        Args:
            raw_batch: 인덱스만 있는 배치
            
        Returns:
            True if submitted successfully
        """
        try:
            # 워커 풀에 배치 처리 제출
            future = self.worker_pool.submit(self._process_single_batch, raw_batch)
            self.processing_futures[future] = {
                'submitted_time': time.time(),
                'batch_size': len(raw_batch)
            }
            
            # 완료된 작업들 체크 및 결과 큐에 추가
            self._check_completed_futures()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to submit batch for processing: {e}")
            return False
    
    def _check_completed_futures(self):
        """완료된 futures 체크하고 ready_queue에 추가"""
        completed_futures = []
        
        for future in list(self.processing_futures.keys()):
            if future.done():
                completed_futures.append(future)
                
        for future in completed_futures:
            try:
                # future가 이미 처리되었는지 확인
                if future not in self.processing_futures:
                    continue
                    
                # 처리 완료된 배치 가져오기
                gpu_ready_batch = future.result()
                batch_info = self.processing_futures.pop(future, None)
                
                if batch_info is None:
                    continue
                
                # 처리 시간 기록
                process_time = time.time() - batch_info['submitted_time']
                with self.stats_lock:
                    self.processing_times.append(process_time)
                    self.processed_count += 1
                
                # Ready 큐에 추가 (blocking 허용 - 더 안정적)
                self.ready_queue.put(gpu_ready_batch, timeout=1.0)
                        
            except queue.Full:
                # 큐가 가득 차면 로깅만
                self.logger.debug(f"Ready queue full, processed batch {self.processed_count}")
            except Exception as e:
                self.error_queue.put(e)
                self.processing_futures.pop(future, None)
    
    def _process_single_batch(self, raw_batch: List[Dict]) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        단일 배치 처리: 인덱싱 → KJT 생성 → pin_memory
        
        Args:
            raw_batch: 인덱스가 포함된 배치
            
        Returns:
            GPU-ready 배치 (CPU에서 KJT 생성 완료, pin_memory 적용)
        """
        start_time = time.time()
        
        # === 1. 인덱싱 단계 ===
        indexing_start = time.time()
        # 1. 인덱스 추출
        notice_indices = [item["notice_idx"] for item in raw_batch]
        company_indices = [item["company_idx"] for item in raw_batch]
        indexing_time = time.time() - indexing_start
        
        # === 2. KJT 생성 단계 ===
        kjt_start = time.time()
        # 2. Notice 데이터 처리 (이미 로딩된 피쳐 사용)
        notice_data = self._process_tower_data(
            indices=notice_indices,
            store=self.dataset.notice_store,
            categorical_keys=self.dataset.schema.notice.categorical,
            tower_name="notice"
        )
        
        # 3. Company 데이터 처리 (이미 로딩된 피쳐 사용)
        company_data = self._process_tower_data(
            indices=company_indices,
            store=self.dataset.company_store,
            categorical_keys=self.dataset.schema.company.categorical,
            tower_name="company"
        )
        kjt_time = time.time() - kjt_start
        
        process_time = time.time() - start_time
        
        # 성능 추적
        with self.stats_lock:
            self.indexing_times.append(indexing_time)
            self.kjt_times.append(kjt_time)
            
            # 매 100개 배치마다 상세 프로파일링 로그
            if self.processed_count % 100 == 0 and len(self.indexing_times) > 0:
                avg_indexing = sum(self.indexing_times[-100:]) / min(100, len(self.indexing_times))
                avg_kjt = sum(self.kjt_times[-100:]) / min(100, len(self.kjt_times))
                
                self.logger.info(
                    f"Batch processing profile (last 100): "
                    f"indexing={avg_indexing:.3f}s, kjt={avg_kjt:.3f}s, "
                    f"total={process_time:.3f}s"
                )
        
        return {
            "notice": notice_data,
            "company": company_data,
            "_meta": {
                "process_time": process_time,
                "batch_size": len(raw_batch),
                "worker_thread": threading.current_thread().name
            }
        }
    
    def _process_tower_data(
        self, 
        indices: List[int], 
        store: Dict,
        categorical_keys: List[str],
        tower_name: str
    ) -> Dict[str, torch.Tensor]:
        """
        단일 타워 데이터 처리
        핵심: 이미 메모리에 있는 피쳐 데이터에서 인덱싱 + KJT 생성
        """
        # 1. Dense 데이터 인덱싱 (이미 projection 완료된 데이터)
        dense_tensor = torch.from_numpy(
            store['dense_projected'][indices]
        ).float()
        
        # 2. Categorical 데이터 인덱싱 + KJT 생성 (최적화)
        categorical_data = store['categorical'][indices]
        
        # 최적화된 KJT 빌더 사용
        kjt = build_kjt_from_numpy(
            categorical_data, 
            categorical_keys,
            pin_memory=False  # 아래에서 처리
        )
        
        # 3. pin_memory 적용 (GPU 전송 가속화)
        if self.pin_memory:
            dense_tensor = dense_tensor.pin_memory()
            # KJT는 GPU 전송 시 pin_memory 처리됨
            
        return {
            "dense": dense_tensor,
            "kjt": kjt
        }
    
    def get_ready_batch(self, timeout: float = 1.0) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        """GPU-ready 배치 가져오기"""
        try:
            # 완료된 futures 다시 체크
            self._check_completed_futures()
            
            # 에러 체크
            if not self.error_queue.empty():
                error = self.error_queue.get_nowait()
                raise error
            
            return self.ready_queue.get(timeout=timeout)
            
        except queue.Empty:
            return None
    
    def get_stats(self) -> Dict[str, float]:
        """처리 통계 반환"""
        with self.stats_lock:
            stats = {
                "processed_batches": self.processed_count,
                "active_workers": len(self.processing_futures),
                "ready_queue_size": self.ready_queue.qsize()
            }
            
            if self.processing_times:
                stats.update({
                    "avg_process_time": sum(self.processing_times) / len(self.processing_times),
                    "min_process_time": min(self.processing_times),
                    "max_process_time": max(self.processing_times)
                })
                
            return stats
    
    def shutdown(self):
        """워커 풀 정리"""
        self.worker_pool.shutdown(wait=True)


class TrueOverlapPipeline:
    """
    진짜 오버랩 파이프라인
    GPU가 배치 N 처리 중 → 배치 N+1,N+2,N+3 백그라운드 처리 완료
    """
    
    def __init__(
        self,
        dataloader: DataLoader,
        device: torch.device,
        num_workers: int = 12,  # 3 -> 12로 대폭 증가
        prefetch_batches: int = 16,  # 4 -> 16으로 증가
        pin_memory: bool = True
    ):
        """
        Args:
            dataloader: lightweight collate_fn 사용하는 DataLoader
            device: GPU 디바이스
            num_workers: 배치 처리 워커 수
            prefetch_batches: 미리 준비할 배치 수
            pin_memory: pin_memory 사용 여부
        """
        self.dataloader = dataloader
        self.device = device
        self.num_workers = num_workers
        self.prefetch_batches = prefetch_batches
        self.pin_memory = pin_memory
        
        # 컴포넌트 초기화
        self.stream_manager = StreamingDataLoaderManager(
            dataloader, 
            buffer_size=prefetch_batches * 2
        )
        
        self.processor_pool = BatchProcessorPool(
            dataset=dataloader.dataset,
            num_workers=num_workers,
            ready_queue_size=prefetch_batches,
            pin_memory=pin_memory
        )
        
        # 제어 스레드
        self.coordinator_thread = None
        self.stop_event = threading.Event()
        
        # 성능 추적
        self.batch_count = 0
        self.gpu_transfer_times = []
        self.total_batches = len(dataloader) if hasattr(dataloader, '__len__') else 0
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)  # INFO 레벨로 변경
    
    def start(self):
        """파이프라인 시작"""
        if self.coordinator_thread is not None:
            self.logger.warning("Pipeline already running")
            return
            
        self.logger.info("Starting True Overlap Pipeline...")
        self.logger.info(f"  - Workers: {self.num_workers}")
        self.logger.info(f"  - Prefetch batches: {self.prefetch_batches}")
        self.logger.info(f"  - Pin memory: {self.pin_memory}")
        
        # 스트림 매니저 시작
        self.stream_manager.start()
        
        # 조정자 스레드 시작
        self.stop_event.clear()
        self.coordinator_thread = threading.Thread(
            target=self._coordination_loop,
            daemon=True,
            name="Pipeline-Coordinator"
        )
        self.coordinator_thread.start()
        
        # 초기 배치들 프리로드
        self._warmup_pipeline()
        
        self.logger.info("True Overlap Pipeline started successfully")
    
    def stop(self):
        """파이프라인 중지"""
        if self.coordinator_thread is None:
            return
            
        self.logger.info("Stopping True Overlap Pipeline...")
        
        self.stop_event.set()
        
        # 조정자 스레드 대기
        if self.coordinator_thread:
            self.coordinator_thread.join(timeout=3.0)
            self.coordinator_thread = None
        
        # 컴포넌트 정리
        self.stream_manager.stop()
        self.processor_pool.shutdown()
        
        self.logger.info("True Overlap Pipeline stopped")
    
    def _coordination_loop(self):
        """조정자 루프: DataLoader → BatchProcessor 연결"""
        coordination_count = 0
        try:
            self.logger.info("Coordination loop started")
            
            while not self.stop_event.is_set():
                # Raw 배치 가져오기 (타임아웃 증가: 1.0 -> 5.0)
                raw_batch = self.stream_manager.get_raw_batch(timeout=5.0)
                
                if raw_batch is None:
                    # 디버깅: 왜 None인지 확인
                    self.logger.debug(f"No raw batch available. Stream queue size: {self.stream_manager.qsize()}, loaded: {self.stream_manager.loaded_count}")
                    continue
                
                # 배치 처리 시작
                success = self.processor_pool.process_batch_async(raw_batch)
                coordination_count += 1
                
                if not success:
                    self.logger.warning("Failed to submit batch for processing")
                else:
                    if coordination_count % 10 == 0:
                        self.logger.debug(f"Coordinated {coordination_count} batches to processor pool")
                    
        except Exception as e:
            self.logger.error(f"Coordination loop error: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
        finally:
            self.logger.info(f"Coordination loop finished. Total coordinated: {coordination_count}")
    
    def _warmup_pipeline(self):
        """파이프라인 워밍업: 초기 배치들 미리 처리 시작"""
        warmup_start = time.time()
        warmup_count = 0
        
        self.logger.info(f"Starting warmup with {self.prefetch_batches} batches...")
        
        for i in range(self.prefetch_batches):
            raw_batch = self.stream_manager.get_raw_batch(timeout=10.0)  # 워밍업은 더 긴 타임아웃
            if raw_batch is None:
                self.logger.warning(f"Warmup: Could not get batch {i+1}/{self.prefetch_batches}")
                self.logger.warning(f"Stream queue size: {self.stream_manager.qsize()}, loaded: {self.stream_manager.loaded_count}")
                break
                
            success = self.processor_pool.process_batch_async(raw_batch)
            if success:
                warmup_count += 1
                self.logger.debug(f"Warmup: Submitted batch {warmup_count}/{self.prefetch_batches}")
            
        warmup_time = time.time() - warmup_start
        self.logger.info(f"Pipeline warmup completed: {warmup_count}/{self.prefetch_batches} batches in {warmup_time:.2f}s")
    
    def get_next_batch(self, timeout: float = 30.0) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        """
        GPU-ready 배치 가져오기 (즉시 사용 가능)
        핵심: non-blocking GPU 전송까지 완료된 상태
        """
        # 1. CPU에서 준비된 배치 가져오기
        cpu_batch = self.processor_pool.get_ready_batch(timeout=timeout)
        
        if cpu_batch is None:
            return None
        
        # 2. GPU로 즉시 전송
        start_time = time.time()
        gpu_batch = self._transfer_to_gpu(cpu_batch)
        transfer_time = time.time() - start_time
        
        # 3. 성능 추적
        self.gpu_transfer_times.append(transfer_time)
        self.batch_count += 1
        
        return gpu_batch
    
    def _transfer_to_gpu(self, cpu_batch: Dict) -> Dict[str, Dict[str, torch.Tensor]]:
        """CPU 배치를 GPU로 전송 (pin_memory로 가속화)"""
        gpu_batch = {
            "notice": self._transfer_tower_data(cpu_batch["notice"]),
            "company": self._transfer_tower_data(cpu_batch["company"])
        }
        
        return gpu_batch
    
    def _transfer_tower_data(self, tower_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """단일 타워 데이터를 GPU로 전송"""
        gpu_tower_data = {}
        
        for key, tensor in tower_data.items():
            if isinstance(tensor, torch.Tensor):
                # pin_memory 적용된 텐서는 non_blocking=True로 빠른 전송
                gpu_tower_data[key] = tensor.to(self.device, non_blocking=self.pin_memory)
            else:
                # KJT의 경우 to() 메서드 사용
                gpu_tower_data[key] = tensor.to(self.device)
                
        return gpu_tower_data
    
    def __len__(self) -> int:
        """총 배치 개수"""
        return self.total_batches
    
    def __iter__(self) -> Iterator[Dict[str, Dict[str, torch.Tensor]]]:
        """Iterator 인터페이스"""
        for _ in range(self.total_batches):
            batch = self.get_next_batch()
            if batch is None:
                break
            yield batch
    
    def get_performance_stats(self) -> Dict[str, float]:
        """성능 통계"""
        stats = {
            "processed_batches": self.batch_count,
            "total_batches": self.total_batches,
            "progress": self.batch_count / self.total_batches if self.total_batches > 0 else 0.0
        }
        
        # GPU 전송 시간
        if self.gpu_transfer_times:
            stats.update({
                "avg_gpu_transfer_time": sum(self.gpu_transfer_times) / len(self.gpu_transfer_times),
                "min_gpu_transfer_time": min(self.gpu_transfer_times),
                "max_gpu_transfer_time": max(self.gpu_transfer_times)
            })
        
        # BatchProcessor 통계
        processor_stats = self.processor_pool.get_stats()
        stats.update({f"processor_{k}": v for k, v in processor_stats.items()})
        
        # StreamManager 통계  
        stats["stream_queue_size"] = self.stream_manager.qsize()
        stats["stream_loaded_count"] = self.stream_manager.loaded_count
        
        return stats
    
    def __del__(self):
        """소멸자에서 정리"""
        self.stop()