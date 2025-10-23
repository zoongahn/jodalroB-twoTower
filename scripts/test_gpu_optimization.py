#!/usr/bin/env python3
"""
GPU 최적화 전후 성능 비교 테스트
배치 크기 256 고정으로 순수 GPU collate 최적화 효과만 측정
"""

import torch
import time
import psutil
import threading
from pathlib import Path
import pandas as pd
import sys

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.database_connector import DatabaseConnector
from src.torchrec_preprocess.schema import build_torchrec_schema_from_meta
from src.towers.pairs.unified_bid_data_loader import (
    create_unified_bid_dataloaders,
    create_collate_fn,
    create_collate_fn_gpu
)


class PerformanceComparator:
    """성능 비교 측정 도구"""

    def __init__(self):
        self.results = {}
        self.is_monitoring = False
        self.monitor_thread = None
        self.gpu_usage = []
        self.cpu_usage = []

    def start_monitoring(self):
        """시스템 모니터링 시작"""
        self.is_monitoring = True
        self.gpu_usage.clear()
        self.cpu_usage.clear()

        def monitor():
            while self.is_monitoring:
                # CPU 사용률
                cpu = psutil.cpu_percent(interval=None)
                self.cpu_usage.append(cpu)

                # GPU 메모리 사용률
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated()
                    reserved = torch.cuda.memory_reserved()
                    if reserved > 0:
                        gpu_mem = (allocated / reserved) * 100
                    else:
                        gpu_mem = 0
                    self.gpu_usage.append(gpu_mem)
                else:
                    self.gpu_usage.append(0)

                time.sleep(0.05)  # 더 자주 측정

        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """시스템 모니터링 종료"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)

    def measure_dataloader_performance(self, test_name: str, dataloader, num_batches: int = 50):
        """DataLoader 성능 측정"""

        print(f"\n🔄 {test_name} 성능 측정 중...")

        self.start_monitoring()

        # 타이밍 측정
        batch_times = []
        gpu_transfer_times = []
        total_start = time.time()

        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            batch_start = time.time()

            # GPU 전송 시뮬레이션 (모델이 하는 작업)
            gpu_start = time.time()
            if torch.cuda.is_available():
                # 데이터가 이미 GPU에 있는지 확인
                notice_gpu = batch["notice"]["dense"]
                company_gpu = batch["company"]["dense"]

                if not notice_gpu.is_cuda:
                    notice_gpu = notice_gpu.cuda(non_blocking=True)
                if not company_gpu.is_cuda:
                    company_gpu = company_gpu.cuda(non_blocking=True)

                # KJT GPU 전송
                notice_kjt = batch["notice"]["kjt"]
                company_kjt = batch["company"]["kjt"]

                if hasattr(notice_kjt, "to") and not notice_kjt.values().is_cuda:
                    notice_kjt = notice_kjt.to("cuda")
                if hasattr(company_kjt, "to") and not company_kjt.values().is_cuda:
                    company_kjt = company_kjt.to("cuda")

                torch.cuda.synchronize()  # GPU 작업 완료 대기

            gpu_end = time.time()

            # 간단한 GPU 연산 (실제 모델 사용 시뮬레이션)
            if torch.cuda.is_available():
                dummy_result = torch.mm(notice_gpu[:32], notice_gpu[:32].T)  # 작은 연산
                torch.cuda.synchronize()

            batch_end = time.time()

            batch_times.append(batch_end - batch_start)
            gpu_transfer_times.append(gpu_end - gpu_start)

            if (i + 1) % 10 == 0:
                print(f"   진행: {i+1}/{num_batches} 배치")

        total_end = time.time()
        self.stop_monitoring()

        # 결과 저장
        self.results[test_name] = {
            'total_time': total_end - total_start,
            'avg_batch_time': sum(batch_times) / len(batch_times),
            'avg_gpu_transfer_time': sum(gpu_transfer_times) / len(gpu_transfer_times),
            'batches_per_second': len(batch_times) / (total_end - total_start),
            'avg_gpu_usage': sum(self.gpu_usage) / len(self.gpu_usage) if self.gpu_usage else 0,
            'max_gpu_usage': max(self.gpu_usage) if self.gpu_usage else 0,
            'avg_cpu_usage': sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0,
            'batch_count': len(batch_times)
        }

        result = self.results[test_name]
        print(f"   완료! 평균 배치 시간: {result['avg_batch_time']*1000:.1f}ms")
        print(f"   배치/초: {result['batches_per_second']:.1f}")
        print(f"   평균 GPU 사용률: {result['avg_gpu_usage']:.1f}%")


def test_cpu_vs_gpu_collate():
    """CPU vs GPU collate 함수 성능 비교"""

    print("🚀 GPU 최적화 전후 성능 비교 테스트")
    print("=" * 60)
    print("조건: 배치 크기 256 고정, collate 함수만 변경")

    # 데이터베이스 연결 및 스키마 구축
    print("\n1️⃣ 테스트 환경 준비...")
    db = DatabaseConnector()
    engine = db.engine

    schema_config = {
        "notice_table": "notice",
        "company_table": "company",
        "pair_table": "bid_two_tower",
        "pair_notice_id_cols": ["bidntceno", "bidntceord"],
        "pair_company_id_cols": ["bizno"],
        "metadata_path": "meta/metadata.csv"
    }
    schema = build_torchrec_schema_from_meta(**schema_config)

    # 공통 설정
    common_config = {
        "db_engine": engine,
        "schema": schema,
        "batch_size": 256,  # 고정
        "test_split": 0.2,
        "shuffle_seed": 42,
        "num_workers": 0,
        "pin_memory": False,
        "streaming": True,
        "load_all_features": True,
        "use_preprocessor": True,
        "test_mode": True,
        "pair_limit": 20000  # 충분한 데이터
    }

    comparator = PerformanceComparator()

    # 테스트 1: 기존 CPU collate 함수
    print("\n2️⃣ 테스트 1: 기존 CPU Collate 함수")

    # 기존 collate 함수 사용을 위해 임시로 함수 덮어쓰기
    import src.towers.pairs.unified_bid_data_loader as dataloader_module
    original_create_collate_fn_gpu = dataloader_module.create_collate_fn_gpu
    dataloader_module.create_collate_fn_gpu = dataloader_module.create_collate_fn

    try:
        train_loader_cpu, _ = create_unified_bid_dataloaders(**common_config)
        comparator.measure_dataloader_performance("CPU_Collate", train_loader_cpu, num_batches=30)
    finally:
        # 원래 함수 복원
        dataloader_module.create_collate_fn_gpu = original_create_collate_fn_gpu

    # GPU 메모리 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 테스트 2: GPU 최적화 collate 함수
    print("\n3️⃣ 테스트 2: GPU 최적화 Collate 함수")
    train_loader_gpu, _ = create_unified_bid_dataloaders(**common_config)
    comparator.measure_dataloader_performance("GPU_Collate", train_loader_gpu, num_batches=30)

    # 결과 분석
    print("\n4️⃣ 성능 비교 결과")
    analyze_results(comparator.results)

    # 결과 저장
    save_results(comparator.results)


def analyze_results(results):
    """결과 분석 및 출력"""

    cpu_result = results["CPU_Collate"]
    gpu_result = results["GPU_Collate"]

    print("=" * 80)
    print("📊 성능 비교 결과")
    print("=" * 80)

    # 배치 처리 시간 비교
    cpu_batch_time = cpu_result['avg_batch_time'] * 1000  # ms
    gpu_batch_time = gpu_result['avg_batch_time'] * 1000  # ms
    batch_improvement = (cpu_batch_time - gpu_batch_time) / cpu_batch_time * 100

    print(f"⏱️  평균 배치 처리 시간:")
    print(f"   CPU Collate: {cpu_batch_time:.1f}ms")
    print(f"   GPU Collate: {gpu_batch_time:.1f}ms")
    print(f"   개선: {batch_improvement:+.1f}% ({'faster' if batch_improvement > 0 else 'slower'})")

    # 처리량 비교
    cpu_throughput = cpu_result['batches_per_second']
    gpu_throughput = gpu_result['batches_per_second']
    throughput_improvement = (gpu_throughput - cpu_throughput) / cpu_throughput * 100

    print(f"\n🚀 처리량 (배치/초):")
    print(f"   CPU Collate: {cpu_throughput:.1f} batches/sec")
    print(f"   GPU Collate: {gpu_throughput:.1f} batches/sec")
    print(f"   개선: {throughput_improvement:+.1f}%")

    # GPU 사용률 비교
    print(f"\n📈 GPU 사용률:")
    print(f"   CPU Collate: 평균 {cpu_result['avg_gpu_usage']:.1f}%, 최대 {cpu_result['max_gpu_usage']:.1f}%")
    print(f"   GPU Collate: 평균 {gpu_result['avg_gpu_usage']:.1f}%, 최대 {gpu_result['max_gpu_usage']:.1f}%")

    gpu_usage_improvement = gpu_result['avg_gpu_usage'] - cpu_result['avg_gpu_usage']
    print(f"   GPU 사용률 증가: {gpu_usage_improvement:+.1f}%p")

    # CPU 사용률 비교
    print(f"\n💻 CPU 사용률:")
    print(f"   CPU Collate: 평균 {cpu_result['avg_cpu_usage']:.1f}%")
    print(f"   GPU Collate: 평균 {gpu_result['avg_cpu_usage']:.1f}%")

    cpu_usage_change = gpu_result['avg_cpu_usage'] - cpu_result['avg_cpu_usage']
    print(f"   CPU 사용률 변화: {cpu_usage_change:+.1f}%p")

    # 전체 요약
    print(f"\n🎯 최적화 효과 요약:")
    if batch_improvement > 10:
        print(f"   ✅ 배치 처리 시간 {batch_improvement:.1f}% 단축")
    elif batch_improvement > 0:
        print(f"   🔄 배치 처리 시간 {batch_improvement:.1f}% 소폭 개선")
    else:
        print(f"   ⚠️ 배치 처리 시간 변화 없음 또는 느려짐")

    if gpu_usage_improvement > 10:
        print(f"   ✅ GPU 사용률 {gpu_usage_improvement:.1f}%p 크게 향상")
    elif gpu_usage_improvement > 0:
        print(f"   🔄 GPU 사용률 {gpu_usage_improvement:.1f}%p 소폭 향상")
    else:
        print(f"   ⚠️ GPU 사용률 개선 없음")

    if throughput_improvement > 20:
        print(f"   🚀 처리량 {throughput_improvement:.1f}% 대폭 개선")
    elif throughput_improvement > 0:
        print(f"   📈 처리량 {throughput_improvement:.1f}% 개선")
    else:
        print(f"   📉 처리량 개선 없음")


def save_results(results):
    """결과를 CSV 파일로 저장"""

    output_dir = Path("profiling_results/gpu_optimization_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 결과를 DataFrame으로 변환
    df_data = []
    for test_name, result in results.items():
        df_data.append({
            'test_name': test_name,
            'total_time': result['total_time'],
            'avg_batch_time_ms': result['avg_batch_time'] * 1000,
            'avg_gpu_transfer_time_ms': result['avg_gpu_transfer_time'] * 1000,
            'batches_per_second': result['batches_per_second'],
            'avg_gpu_usage_percent': result['avg_gpu_usage'],
            'max_gpu_usage_percent': result['max_gpu_usage'],
            'avg_cpu_usage_percent': result['avg_cpu_usage'],
            'batch_count': result['batch_count']
        })

    df = pd.DataFrame(df_data)
    df.to_csv(output_dir / "gpu_optimization_comparison.csv", index=False)

    print(f"\n📁 결과 저장: {output_dir}/gpu_optimization_comparison.csv")

    # 개선율 계산 및 저장
    if len(results) == 2:
        cpu_result = results["CPU_Collate"]
        gpu_result = results["GPU_Collate"]

        improvements = {
            'batch_time_improvement_percent': ((cpu_result['avg_batch_time'] - gpu_result['avg_batch_time']) / cpu_result['avg_batch_time'] * 100),
            'throughput_improvement_percent': ((gpu_result['batches_per_second'] - cpu_result['batches_per_second']) / cpu_result['batches_per_second'] * 100),
            'gpu_usage_improvement_pp': (gpu_result['avg_gpu_usage'] - cpu_result['avg_gpu_usage']),
            'cpu_usage_change_pp': (gpu_result['avg_cpu_usage'] - cpu_result['avg_cpu_usage'])
        }

        improvement_df = pd.DataFrame([improvements])
        improvement_df.to_csv(output_dir / "optimization_improvements.csv", index=False)


if __name__ == "__main__":
    test_cpu_vs_gpu_collate()