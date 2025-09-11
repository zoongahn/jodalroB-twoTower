"""
Async Batch Preprocessor for Two-Tower Training
배치별 KJT 생성을 백그라운드에서 미리 처리하여 GPU 대기시간 제거
"""
import torch
import threading
import queue
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from src.towers.pairs.unified_bid_data_loader import _build_batch_kjt


@dataclass
class PreprocessedBatch:
    """전처리 완료된 배치 데이터"""
    notice_data: Dict[str, torch.Tensor]
    company_data: Dict[str, torch.Tensor]
    batch_info: Dict[str, Any]
    prep_time: float


class AsyncBatchPreprocessor:
    """
    배치별 인덱싱 → KJT 생성을 백그라운드에서 미리 처리
    
    Pipeline:
    1. Raw batch (indices) → Index Queue
    2. Background workers: Index → KJT (CPU + pin_memory)
    3. GPU-ready batch → Ready Queue
    4. GPU thread: immediate non_blocking transfer
    """
    
    def __init__(
        self,
        dataset,
        prefetch_size: int = 3,
        num_workers: int = 2,
        pin_memory: bool = True
    ):
        """
        Args:
            dataset: UnifiedBidDataset with pre-loaded features
            prefetch_size: Number of batches to prepare in advance
            num_workers: Number of background worker threads
            pin_memory: Apply pin_memory for faster GPU transfer
        """
        self.dataset = dataset
        self.prefetch_size = prefetch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # 2-stage queues
        self.index_queue = queue.Queue(maxsize=prefetch_size * 2)  # Raw batches with indices
        self.ready_queue = queue.Queue(maxsize=prefetch_size)      # GPU-ready batches
        self.error_queue = queue.Queue()
        
        # Threading control
        self.stop_event = threading.Event()
        self.workers = []
        self.stats_lock = threading.Lock()
        
        # Performance tracking
        self.total_batches = 0
        self.prep_times = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def start(self):
        """Start background preprocessing workers"""
        if self.workers:
            self.logger.warning("Workers already running")
            return
            
        self.stop_event.clear()
        
        # Start worker threads
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                args=(i,),
                daemon=True,
                name=f"BatchPreprocessor-{i}"
            )
            worker.start()
            self.workers.append(worker)
            
        self.logger.info(f"Started {self.num_workers} preprocessing workers")
    
    def stop(self):
        """Stop all background workers"""
        if not self.workers:
            return
            
        self.stop_event.set()
        
        # Clear queues to unblock workers
        self._clear_queues()
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=2.0)
            
        self.workers.clear()
        self.logger.info("All preprocessing workers stopped")
    
    def submit_batch(self, raw_batch: List[Dict], timeout: float = 1.0) -> bool:
        """
        Submit raw batch for preprocessing
        
        Args:
            raw_batch: List of batch items with indices
            timeout: Queue timeout
            
        Returns:
            True if submitted successfully, False if queue full
        """
        try:
            self.index_queue.put(raw_batch, timeout=timeout)
            return True
        except queue.Full:
            self.logger.warning("Index queue full, dropping batch")
            return False
    
    def get_ready_batch(self, timeout: float = 30.0) -> Optional[PreprocessedBatch]:
        """
        Get preprocessed batch ready for GPU
        
        Args:
            timeout: Maximum wait time
            
        Returns:
            PreprocessedBatch or None if timeout/error
        """
        try:
            # Check for errors first
            if not self.error_queue.empty():
                error = self.error_queue.get_nowait()
                raise error
                
            # Get ready batch
            batch = self.ready_queue.get(timeout=timeout)
            return batch
            
        except queue.Empty:
            self.logger.warning("Ready queue timeout - preprocessing slower than consumption")
            return None
    
    def _worker_loop(self, worker_id: int):
        """Main worker loop for batch preprocessing"""
        self.logger.info(f"Worker {worker_id} started")
        
        try:
            while not self.stop_event.is_set():
                try:
                    # Get raw batch
                    raw_batch = self.index_queue.get(timeout=1.0)
                    
                    if raw_batch is None:  # Stop signal
                        break
                        
                    # Process batch
                    start_time = time.time()
                    preprocessed = self._preprocess_batch(raw_batch, worker_id)
                    prep_time = time.time() - start_time
                    
                    # Add timing info
                    preprocessed.prep_time = prep_time
                    
                    # Put in ready queue
                    if not self.stop_event.is_set():
                        self.ready_queue.put(preprocessed)
                        
                    # Update stats
                    with self.stats_lock:
                        self.prep_times.append(prep_time)
                        self.total_batches += 1
                        
                except queue.Empty:
                    continue
                    
        except Exception as e:
            self.error_queue.put(e)
            self.logger.error(f"Worker {worker_id} error: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
        finally:
            self.logger.info(f"Worker {worker_id} finished")
    
    def _preprocess_batch(self, raw_batch: List[Dict], worker_id: int) -> PreprocessedBatch:
        """
        Core preprocessing: indices → GPU-ready KJT
        
        Args:
            raw_batch: List of items with notice_idx, company_idx
            worker_id: Worker thread ID for debugging
            
        Returns:
            PreprocessedBatch with GPU-ready tensors
        """
        # Extract indices
        notice_indices = [item["notice_idx"] for item in raw_batch]
        company_indices = [item["company_idx"] for item in raw_batch]
        
        # Preprocess notice data
        notice_data = self._preprocess_tower_data(
            indices=notice_indices,
            store=self.dataset.notice_store,
            categorical_keys=self.dataset.schema.notice.categorical,
            tower_name="notice"
        )
        
        # Preprocess company data
        company_data = self._preprocess_tower_data(
            indices=company_indices,
            store=self.dataset.company_store,
            categorical_keys=self.dataset.schema.company.categorical,
            tower_name="company"
        )
        
        return PreprocessedBatch(
            notice_data=notice_data,
            company_data=company_data,
            batch_info={
                "batch_size": len(raw_batch),
                "worker_id": worker_id,
                "notice_indices": notice_indices[:5],  # Sample for debugging
                "company_indices": company_indices[:5]
            },
            prep_time=0.0  # Will be set by caller
        )
    
    def _preprocess_tower_data(
        self, 
        indices: List[int], 
        store: Dict,
        categorical_keys: List[str],
        tower_name: str
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess single tower data (notice or company)
        
        Returns:
            Dict with 'dense' and 'kjt' tensors (CPU, pinned if requested)
        """
        # Dense data processing
        dense_tensor = torch.from_numpy(
            store['dense_projected'][indices]
        ).float()
        
        # Categorical data processing
        categorical_data = store['categorical'][indices]
        categorical_tensor = torch.from_numpy(categorical_data).long()
        
        # Create KJT on CPU
        kjt = _build_batch_kjt(categorical_tensor, categorical_keys)
        
        # Apply pin_memory for faster GPU transfer
        if self.pin_memory:
            dense_tensor = dense_tensor.pin_memory()
            # Note: KJT pin_memory is handled differently - will be done during GPU transfer
            
        return {
            "dense": dense_tensor,
            "kjt": kjt
        }
    
    def _clear_queues(self):
        """Clear all queues to unblock workers"""
        # Clear index queue
        while not self.index_queue.empty():
            try:
                self.index_queue.get_nowait()
            except queue.Empty:
                break
                
        # Clear ready queue
        while not self.ready_queue.empty():
            try:
                self.ready_queue.get_nowait()
            except queue.Empty:
                break
                
        # Send stop signals
        for _ in range(self.num_workers):
            try:
                self.index_queue.put(None, timeout=0.1)
            except queue.Full:
                break
    
    def get_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        with self.stats_lock:
            if not self.prep_times:
                return {"total_batches": self.total_batches}
                
            return {
                "total_batches": self.total_batches,
                "avg_prep_time": sum(self.prep_times) / len(self.prep_times),
                "min_prep_time": min(self.prep_times),
                "max_prep_time": max(self.prep_times),
                "index_queue_size": self.index_queue.qsize(),
                "ready_queue_size": self.ready_queue.qsize()
            }
    
    def __del__(self):
        """Cleanup on destruction"""
        self.stop()