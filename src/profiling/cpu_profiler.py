#!/usr/bin/env python3
"""
CPU 병목 지점 검출을 위한 프로파일링 도구
GPU 점유율이 낮고 배치 처리 속도가 느린 문제 해결용
"""

import torch
import time
import psutil
import threading
from typing import Dict, List, Optional, Callable
from contextlib import contextmanager
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler
import pandas as pd
from pathlib import Path


class CPUBottleneckProfiler:
    """
    CPU 병목 지점 검출 및 분석을 위한 프로파일러
    데이터 파이프라인의 각 단계별 성능 측정
    """

    def __init__(self, output_dir: str = "profiling_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.timings = {}
        self.gpu_utilization = []
        self.cpu_utilization = []
        self.memory_usage = []
        self.is_monitoring = False
        self.monitor_thread = None

    @contextmanager
    def profile_batch_processing(self, num_batches: int = 10, warmup_batches: int = 3):
        """
        배치 처리 과정의 세부 프로파일링

        Args:
            num_batches: 프로파일링할 배치 수
            warmup_batches: 워밍업 배치 수 (프로파일링에서 제외)
        """
        print(f"🔍 배치 처리 프로파일링 시작 (warmup: {warmup_batches}, profile: {num_batches})")

        # PyTorch Profiler 설정
        profiler_schedule = schedule(
            wait=warmup_batches,
            warmup=1,
            active=num_batches,
            repeat=1
        )

        # PyTorch 버전 호환성을 위해 간단한 프로파일러 사용
        try:
            prof_context = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )
            prof = prof_context.__enter__()
        except Exception as e:
            print(f"⚠️ PyTorch Profiler 초기화 실패: {e}")
            prof = None
            prof_context = None
        # 시스템 모니터링 시작
        self.start_system_monitoring()

        try:
            yield prof
        finally:
            # 시스템 모니터링 종료
            self.stop_system_monitoring()

            # PyTorch Profiler 정리
            if prof_context:
                try:
                    prof_context.__exit__(None, None, None)
                except:
                    pass

        # 프로파일링 결과 저장
        if prof:
            self._save_profiling_results(prof)
        print(f"✅ 프로파일링 완료! 결과 저장: {self.output_dir}")

    @contextmanager
    def time_operation(self, operation_name: str):
        """특정 작업의 실행 시간 측정"""
        start_time = time.time()

        with record_function(operation_name):
            try:
                yield
            finally:
                end_time = time.time()
                elapsed = end_time - start_time

                if operation_name not in self.timings:
                    self.timings[operation_name] = []
                self.timings[operation_name].append(elapsed)

    def start_system_monitoring(self, interval: float = 0.1):
        """시스템 리소스 모니터링 시작"""
        self.is_monitoring = True
        self.gpu_utilization.clear()
        self.cpu_utilization.clear()
        self.memory_usage.clear()

        def monitor():
            while self.is_monitoring:
                # CPU 사용률
                cpu_percent = psutil.cpu_percent(interval=None)
                self.cpu_utilization.append(cpu_percent)

                # 메모리 사용률
                memory = psutil.virtual_memory()
                self.memory_usage.append(memory.percent)

                # GPU 사용률 (간단한 메모리 사용률로 대체)
                try:
                    if torch.cuda.is_available():
                        # GPU 메모리 사용률 계산 (ZeroDivisionError 방지)
                        allocated = torch.cuda.memory_allocated()
                        reserved = torch.cuda.memory_reserved()
                        if reserved > 0:
                            gpu_memory = (allocated / reserved) * 100
                        else:
                            gpu_memory = 0
                        self.gpu_utilization.append(gpu_memory)
                    else:
                        self.gpu_utilization.append(0)
                except Exception as e:
                    # GPU 모니터링 실패 시 0으로 설정
                    self.gpu_utilization.append(0)

                time.sleep(interval)

        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()

    def stop_system_monitoring(self):
        """시스템 리소스 모니터링 종료"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)

    def _save_profiling_results(self, prof):
        """프로파일링 결과 저장 및 분석"""

        # 1. PyTorch Profiler 테이블 저장 (prof가 있는 경우만)
        if prof:
            try:
                cpu_table = prof.key_averages().table(sort_by="cpu_time_total", row_limit=20)
                with open(self.output_dir / "cpu_bottlenecks.txt", "w") as f:
                    f.write("=== CPU 병목 지점 Top 20 ===\n\n")
                    f.write(cpu_table)

                cuda_table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=20)
                with open(self.output_dir / "cuda_performance.txt", "w") as f:
                    f.write("=== CUDA 성능 분석 Top 20 ===\n\n")
                    f.write(cuda_table)
            except Exception as e:
                print(f"⚠️ PyTorch Profiler 결과 저장 실패: {e}")
        else:
            # Profiler가 없는 경우 기본 메시지 저장
            with open(self.output_dir / "profiling_info.txt", "w") as f:
                f.write("PyTorch Profiler를 사용할 수 없음. 커스텀 타이밍 결과만 제공됩니다.\n")

        # 2. 커스텀 타이밍 결과 저장
        timing_results = []
        for operation, times in self.timings.items():
            timing_results.append({
                'operation': operation,
                'count': len(times),
                'total_time': sum(times),
                'avg_time': sum(times) / len(times),
                'min_time': min(times),
                'max_time': max(times)
            })

        timing_df = pd.DataFrame(timing_results)
        timing_df = timing_df.sort_values('total_time', ascending=False)
        timing_df.to_csv(self.output_dir / "operation_timings.csv", index=False)

        # 3. 시스템 리소스 사용률 저장
        if self.gpu_utilization and self.cpu_utilization:
            resource_df = pd.DataFrame({
                'gpu_utilization': self.gpu_utilization,
                'cpu_utilization': self.cpu_utilization,
                'memory_usage': self.memory_usage
            })
            resource_df.to_csv(self.output_dir / "system_utilization.csv", index=False)

            # 평균 사용률 출력
            print(f"📊 평균 GPU 사용률: {sum(self.gpu_utilization)/len(self.gpu_utilization):.1f}%")
            print(f"📊 평균 CPU 사용률: {sum(self.cpu_utilization)/len(self.cpu_utilization):.1f}%")

        # 4. 병목 지점 분석 리포트
        self._generate_bottleneck_report()

    def _generate_bottleneck_report(self):
        """병목 지점 분석 리포트 생성"""
        report = []
        report.append("=== CPU 병목 지점 분석 리포트 ===\n")

        # GPU 사용률 분석
        if self.gpu_utilization:
            avg_gpu = sum(self.gpu_utilization) / len(self.gpu_utilization)
            if avg_gpu < 50:
                report.append(f"⚠️  GPU 사용률이 낮음: {avg_gpu:.1f}% (목표: >80%)")
                report.append("   → CPU 병목으로 인해 GPU가 대기 상태")
            else:
                report.append(f"✅ GPU 사용률 양호: {avg_gpu:.1f}%")

        # 커스텀 타이밍 분석
        if self.timings:
            report.append("\n=== 주요 병목 작업 (상위 5개) ===")
            sorted_ops = sorted(self.timings.items(),
                              key=lambda x: sum(x[1]), reverse=True)

            for i, (op, times) in enumerate(sorted_ops[:5]):
                total_time = sum(times)
                avg_time = total_time / len(times)
                report.append(f"{i+1}. {op}")
                report.append(f"   총 시간: {total_time:.3f}s, 평균: {avg_time:.3f}s, 호출: {len(times)}회")

                # 병목 지점별 최적화 제안
                if "collate" in op.lower():
                    report.append("   💡 최적화 제안: GPU collate, 비동기 처리, 더 작은 배치 크기")
                elif "database" in op.lower() or "sql" in op.lower():
                    report.append("   💡 최적화 제안: 커넥션 풀링, 배치 쿼리, 인덱스 최적화")
                elif "numpy" in op.lower() or "tensor" in op.lower():
                    report.append("   💡 최적화 제안: pin_memory, 비동기 GPU 전송, 텐서 재사용")
                elif "kjt" in op.lower():
                    report.append("   💡 최적화 제안: GPU에서 직접 KJT 생성, 메모리 사전 할당")

        # 리포트 저장
        with open(self.output_dir / "bottleneck_analysis.txt", "w") as f:
            f.write("\n".join(report))

        # 콘솔에도 출력
        print("\n".join(report))


def profile_dataloader_performance(dataloader, profiler: CPUBottleneckProfiler, num_batches: int = 10):
    """
    DataLoader 성능 프로파일링
    타워 입력 전까지의 전체 파이프라인 분석
    """

    with profiler.profile_batch_processing(num_batches=num_batches, warmup_batches=3):

        for batch_idx, batch in enumerate(dataloader):

            # 배치 전체 처리 시간
            with profiler.time_operation("total_batch_processing"):

                # 1. 데이터 타입 확인
                with profiler.time_operation("batch_inspection"):
                    notice_shape = batch["notice"]["dense"].shape if "notice" in batch else None
                    company_shape = batch["company"]["dense"].shape if "company" in batch else None

                # 2. GPU 전송 시뮬레이션 (실제 모델이 하는 작업)
                with profiler.time_operation("gpu_transfer"):
                    if torch.cuda.is_available():
                        notice_gpu = batch["notice"]["dense"].cuda(non_blocking=True)
                        company_gpu = batch["company"]["dense"].cuda(non_blocking=True)

                        # KJT GPU 전송
                        if hasattr(batch["notice"]["kjt"], "to"):
                            notice_kjt_gpu = batch["notice"]["kjt"].to("cuda")
                        if hasattr(batch["company"]["kjt"], "to"):
                            company_kjt_gpu = batch["company"]["kjt"].to("cuda")

                # 3. 간단한 GPU 연산 (타워 입력 시뮬레이션)
                with profiler.time_operation("simple_gpu_computation"):
                    if torch.cuda.is_available():
                        # 간단한 행렬 곱셈으로 GPU 작업 시뮬레이션
                        dummy_result = torch.mm(notice_gpu, notice_gpu.T)
                        torch.cuda.synchronize()  # GPU 연산 완료 대기

            # 프로파일러 step 호출 (PyTorch Profiler용)
            # PyTorch 1.x 호환성을 위해 step 호출 방식 수정
            pass  # step은 context manager에서 자동 처리됨

            if batch_idx >= num_batches + 2:  # warmup 포함
                break

        print(f"✅ {num_batches}개 배치 프로파일링 완료")


def profile_collate_function_separately(dataset, batch_size: int = 256):
    """
    Collate 함수만 별도로 프로파일링
    가장 의심되는 CPU 병목 지점
    """

    profiler = CPUBottleneckProfiler(output_dir="profiling_results/collate_only")

    # 샘플 배치 데이터 준비 (Dataset.__getitem__ 호출)
    print("📦 샘플 배치 데이터 준비 중...")
    sample_batch = []

    with profiler.time_operation("dataset_getitem_batch"):
        for i in range(batch_size):
            with profiler.time_operation("dataset_getitem_single"):
                item = dataset[i % len(dataset)]
                sample_batch.append(item)

    # Collate 함수 가져오기
    from src.towers.pairs.unified_bid_data_loader import create_collate_fn
    collate_fn = create_collate_fn(dataset)

    print(f"🔧 Collate 함수 프로파일링 (배치 크기: {batch_size})...")

    # Collate 함수 단독 프로파일링
    with profiler.profile_batch_processing(num_batches=5, warmup_batches=1):
        for i in range(7):  # warmup + 5 batches

            with profiler.time_operation("collate_function_total"):

                # 개별 단계별 측정
                with profiler.time_operation("collate_index_extraction"):
                    notice_indices = [item["notice_idx"] for item in sample_batch]
                    company_indices = [item["company_idx"] for item in sample_batch]

                with profiler.time_operation("collate_numpy_access"):
                    # NumPy 배열 접근 시뮬레이션
                    notice_dense_np = dataset.notice_store['dense_projected'][notice_indices]
                    company_dense_np = dataset.company_store['dense_projected'][company_indices]

                with profiler.time_operation("collate_numpy_to_torch"):
                    # NumPy → PyTorch 변환
                    notice_dense = torch.from_numpy(notice_dense_np).float()
                    company_dense = torch.from_numpy(company_dense_np).float()

                with profiler.time_operation("collate_kjt_creation"):
                    # KJT 생성
                    notice_cat = dataset.notice_store['categorical'][notice_indices]
                    company_cat = dataset.company_store['categorical'][company_indices]

                    from src.towers.kjt_utils import create_kjt_from_batch_gpu
                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                    notice_kjt = create_kjt_from_batch_gpu(
                        torch.from_numpy(notice_cat).long().to(device),
                        dataset.schema.notice.categorical,
                        device
                    )
                    company_kjt = create_kjt_from_batch_gpu(
                        torch.from_numpy(company_cat).long().to(device),
                        dataset.schema.company.categorical,
                        device
                    )

            # PyTorch 1.x 호환성을 위해 step 호출 제거
            pass

    print("✅ Collate 함수 프로파일링 완료")


# 사용 예제 함수
def run_comprehensive_profiling():
    """
    종합적인 CPU 병목 분석 실행
    train.py에서 호출하여 사용
    """

    print("🔍 종합 CPU 병목 분석 시작...")

    # 기본 설정 (train.py와 동일)
    from database.database_connector import DatabaseConnector
    from preprocess.torchrec.schema import build_torchrec_schema_from_meta
    from src.towers.pairs.unified_bid_data_loader import create_unified_bid_dataloaders

    db = DatabaseConnector()
    engine = db.engine

    # 스키마 설정 (.env에서 자동으로 설정 읽음)
    schema_config = {
        "pair_notice_id_cols": ["bidntceno", "bidntceord"],
        "pair_company_id_cols": ["bizno"],
        "metadata_path": "meta/metadata.csv"
    }
    schema = build_torchrec_schema_from_meta(**schema_config)

    # 테스트 모드로 DataLoader 생성 (빠른 분석)
    train_loader, _ = create_unified_bid_dataloaders(
        db_engine=engine,
        schema=schema,
        batch_size=256,  # 실제 사용하는 배치 크기
        test_split=0.2,
        shuffle_seed=42,
        num_workers=0,
        pin_memory=False,
        streaming=True,
        load_all_features=True,
        chunk_size=1000000,
        feature_chunksize=1000,
        use_preprocessor=True,
        test_mode=True,
        pair_limit=10000  # 빠른 테스트를 위해 제한
    )

    # 1. 전체 DataLoader 프로파일링
    print("\n1️⃣ 전체 DataLoader 파이프라인 프로파일링...")
    main_profiler = CPUBottleneckProfiler(output_dir="profiling_results/full_pipeline")
    profile_dataloader_performance(train_loader, main_profiler, num_batches=10)

    # 2. Collate 함수 단독 프로파일링
    print("\n2️⃣ Collate 함수 단독 프로파일링...")
    profile_collate_function_separately(train_loader.dataset, batch_size=256)

    print("\n🎯 종합 분석 완료! 결과 확인:")
    print("   - profiling_results/full_pipeline/")
    print("   - profiling_results/collate_only/")
    print("   - TensorBoard: tensorboard --logdir profiling_results/full_pipeline/torch_trace")


if __name__ == "__main__":
    # 테스트 실행
    run_comprehensive_profiling()