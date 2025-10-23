#!/usr/bin/env python3
"""
간단한 CPU 병목 검출 스크립트
PyTorch Profiler 없이 직접 타이밍 측정으로 병목 지점 찾기
"""

import torch
import time
import psutil
import threading
from pathlib import Path
import pandas as pd
from typing import Dict, List
import sys

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.database_connector import DatabaseConnector
from src.torchrec_preprocess.schema import build_torchrec_schema_from_meta
from src.towers.pairs.unified_bid_data_loader import create_unified_bid_dataloaders


class SimpleCPUProfiler:
    """간단한 CPU 병목 검출기"""

    def __init__(self):
        self.timings = {}
        self.gpu_memory_usage = []
        self.cpu_usage = []
        self.is_monitoring = False
        self.monitor_thread = None

    def start_monitoring(self):
        """시스템 모니터링 시작"""
        self.is_monitoring = True
        self.gpu_memory_usage.clear()
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
                    self.gpu_memory_usage.append(gpu_mem)
                else:
                    self.gpu_memory_usage.append(0)

                time.sleep(0.1)

        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """시스템 모니터링 종료"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)

    def time_it(self, name: str, func, *args, **kwargs):
        """함수 실행 시간 측정"""
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()

        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(end - start)

        return result

    def get_results(self):
        """결과 분석 및 반환"""
        results = {}

        # 타이밍 결과
        for operation, times in self.timings.items():
            results[operation] = {
                'total_time': sum(times),
                'avg_time': sum(times) / len(times),
                'count': len(times),
                'min_time': min(times),
                'max_time': max(times)
            }

        # 시스템 사용률
        if self.gpu_memory_usage:
            results['system'] = {
                'avg_gpu_memory': sum(self.gpu_memory_usage) / len(self.gpu_memory_usage),
                'avg_cpu': sum(self.cpu_usage) / len(self.cpu_usage),
                'max_gpu_memory': max(self.gpu_memory_usage),
                'max_cpu': max(self.cpu_usage)
            }

        return results


def profile_dataloader_bottlenecks():
    """DataLoader의 주요 병목 지점들을 직접 측정"""

    print("🔍 간단한 CPU 병목 검출 시작...")

    profiler = SimpleCPUProfiler()

    # 1. 데이터베이스 연결
    print("1️⃣ 데이터베이스 연결...")
    db = profiler.time_it("database_connection", DatabaseConnector)
    engine = db.engine

    # 2. 스키마 구축
    print("2️⃣ 스키마 구축...")
    schema_config = {
        "notice_table": "notice",
        "company_table": "company",
        "pair_table": "bid_two_tower",
        "pair_notice_id_cols": ["bidntceno", "bidntceord"],
        "pair_company_id_cols": ["bizno"],
        "metadata_path": "meta/metadata.csv"
    }
    schema = profiler.time_it("schema_building",
                             build_torchrec_schema_from_meta, **schema_config)

    # 3. DataLoader 생성
    print("3️⃣ DataLoader 생성...")
    train_loader, test_loader = profiler.time_it("dataloader_creation",
        create_unified_bid_dataloaders,
        db_engine=engine,
        schema=schema,
        batch_size=256,
        test_split=0.2,
        shuffle_seed=42,
        num_workers=0,
        pin_memory=False,
        streaming=True,
        load_all_features=True,
        use_preprocessor=True,
        test_mode=True,
        pair_limit=10000  # 빠른 테스트
    )

    print(f"   Train 배치: {len(train_loader)}")

    # 4. 배치 처리 분석
    print("4️⃣ 배치 처리 병목 분석...")
    profiler.start_monitoring()

    batch_count = 0
    max_batches = 20

    for batch in train_loader:
        if batch_count >= max_batches:
            break

        # 배치 데이터 접근 시간
        start_batch = time.time()

        # GPU 전송 시간 측정
        gpu_start = time.time()
        if torch.cuda.is_available():
            notice_gpu = batch["notice"]["dense"].cuda(non_blocking=True)
            company_gpu = batch["company"]["dense"].cuda(non_blocking=True)

            # KJT GPU 전송
            if hasattr(batch["notice"]["kjt"], "to"):
                notice_kjt_gpu = batch["notice"]["kjt"].to("cuda")
            if hasattr(batch["company"]["kjt"], "to"):
                company_kjt_gpu = batch["company"]["kjt"].to("cuda")
        gpu_end = time.time()

        # GPU 연산 시뮬레이션
        compute_start = time.time()
        if torch.cuda.is_available():
            # 간단한 행렬 곱셈
            dummy_result = torch.mm(notice_gpu, notice_gpu.T)
            torch.cuda.synchronize()
        compute_end = time.time()

        batch_end = time.time()

        # 개별 타이밍 기록
        profiler.timings.setdefault("batch_total", []).append(batch_end - start_batch)
        profiler.timings.setdefault("gpu_transfer", []).append(gpu_end - gpu_start)
        profiler.timings.setdefault("gpu_compute", []).append(compute_end - compute_start)

        batch_count += 1

        if batch_count % 5 == 0:
            print(f"   진행: {batch_count}/{max_batches} 배치")

    profiler.stop_monitoring()

    # 5. 결과 분석
    print("\n5️⃣ 결과 분석...")
    results = profiler.get_results()

    print("\n=== CPU 병목 분석 결과 ===")

    # 주요 병목 지점 출력
    timing_items = [(name, data['total_time']) for name, data in results.items()
                   if isinstance(data, dict) and 'total_time' in data]
    timing_items.sort(key=lambda x: x[1], reverse=True)

    print("\n🎯 시간이 오래 걸리는 작업 순위:")
    for i, (operation, total_time) in enumerate(timing_items, 1):
        avg_time = results[operation]['avg_time']
        count = results[operation]['count']
        print(f"{i}. {operation}")
        print(f"   총 시간: {total_time:.3f}s, 평균: {avg_time:.3f}s, 호출: {count}회")

        # 병목 원인 분석
        if "dataloader" in operation.lower():
            print("   💡 DataLoader 생성이 오래 걸림 → 피처 로딩 최적화 필요")
        elif "batch" in operation.lower():
            print("   💡 배치 처리가 오래 걸림 → Collate 함수 최적화 필요")
        elif "gpu_transfer" in operation.lower():
            print("   💡 GPU 전송이 오래 걸림 → 비동기 전송 또는 GPU 직접 생성 필요")
        elif "database" in operation.lower():
            print("   💡 DB 연결이 오래 걸림 → 커넥션 풀링 필요")

    # 시스템 사용률
    if 'system' in results:
        sys_info = results['system']
        print(f"\n📊 시스템 사용률:")
        print(f"   평균 GPU 메모리: {sys_info['avg_gpu_memory']:.1f}%")
        print(f"   평균 CPU 사용률: {sys_info['avg_cpu']:.1f}%")
        print(f"   최대 GPU 메모리: {sys_info['max_gpu_memory']:.1f}%")
        print(f"   최대 CPU 사용률: {sys_info['max_cpu']:.1f}%")

        # GPU 사용률 분석
        if sys_info['avg_gpu_memory'] < 30:
            print("   ⚠️ GPU 메모리 사용률이 낮음 → CPU 병목 의심")
        elif sys_info['avg_gpu_memory'] > 80:
            print("   ✅ GPU 메모리 사용률 양호")

    # 최적화 권장사항
    print("\n🚀 즉시 적용 가능한 최적화:")
    print("1. 배치 크기 증가 (256 → 512)")
    print("2. GPU에서 직접 KJT 생성 (collate_fn 수정)")
    print("3. 비동기 데이터 로딩 (num_workers=2~4)")
    print("4. Database 쿼리 최적화 (커넥션 풀링)")

    # 결과를 CSV로 저장
    output_dir = Path("profiling_results/simple_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    timing_df = pd.DataFrame([
        {
            'operation': name,
            'total_time': data['total_time'],
            'avg_time': data['avg_time'],
            'count': data['count']
        }
        for name, data in results.items()
        if isinstance(data, dict) and 'total_time' in data
    ])
    timing_df.to_csv(output_dir / "timing_analysis.csv", index=False)

    print(f"\n📁 결과 저장: {output_dir}/timing_analysis.csv")

    return results


if __name__ == "__main__":
    profile_dataloader_bottlenecks()