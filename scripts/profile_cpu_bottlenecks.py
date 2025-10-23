#!/usr/bin/env python3
"""
독립적인 CPU 병목 검출 스크립트
학습 실행 없이 데이터 파이프라인만 분석하여 CPU 병목 지점 검출

사용법:
python scripts/profile_cpu_bottlenecks.py
"""

import torch
import sys
import os
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.profiling.cpu_profiler import CPUBottleneckProfiler, profile_dataloader_performance, profile_collate_function_separately
from data.database_connector import DatabaseConnector
from src.torchrec_preprocess.schema import build_torchrec_schema_from_meta
from src.towers.pairs.unified_bid_data_loader import create_unified_bid_dataloaders


def main():
    """
    메인 프로파일링 실행 함수
    GPU 점유율이 40%밖에 안 되는 문제 분석
    """

    print("🔍 CPU 병목 지점 검출 시작...")
    print("목표: GPU 사용률을 40% → 80%+ 향상")
    print("=" * 60)

    # CUDA 설정 확인
    if torch.cuda.is_available():
        print(f"✅ CUDA 사용 가능: {torch.cuda.get_device_name(0)}")
        print(f"   GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        print("⚠️  CUDA 사용 불가 - CPU 모드로 실행")

    # 1. 데이터베이스 연결 및 스키마 구축
    print("\n1️⃣ 데이터베이스 연결 중...")
    db = DatabaseConnector()
    engine = db.engine

    # 스키마 설정 (.env에서 자동으로 설정 읽음)
    schema_config = {
        "pair_notice_id_cols": ["bidntceno", "bidntceord"],
        "pair_company_id_cols": ["bizno"],
        "metadata_path": "meta/metadata.csv"
    }

    schema = build_torchrec_schema_from_meta(**schema_config)
    print(f"   Notice 피처: {len(schema.notice.categorical)}개 범주형, {len(schema.notice.numeric)}개 수치형")
    print(f"   Company 피처: {len(schema.company.categorical)}개 범주형, {len(schema.company.numeric)}개 수치형")

    # 2. 테스트 설정들
    test_configs = [
        {
            "name": "현재 설정 (실제 배치 크기)",
            "batch_size": 256,
            "pair_limit": 50000,
            "test_split": 0.2,
            "streaming": True,
            "load_all_features": True,
            "use_preprocessor": True,
            "num_workers": 0,
            "pin_memory": False
        },
        {
            "name": "GPU 최적화 설정",
            "batch_size": 512,  # 더 큰 배치
            "pair_limit": 50000,
            "test_split": 0.2,
            "streaming": True,
            "load_all_features": True,
            "use_preprocessor": True,
            "num_workers": 0,
            "pin_memory": False
        },
        {
            "name": "멀티프로세싱 설정",
            "batch_size": 256,
            "pair_limit": 50000,
            "test_split": 0.2,
            "streaming": True,
            "load_all_features": True,
            "use_preprocessor": True,
            "num_workers": 4,  # 멀티프로세싱
            "pin_memory": True
        }
    ]

    # 3. 각 설정별 프로파일링
    for i, config in enumerate(test_configs):
        print(f"\n{i+2}️⃣ {config['name']} 프로파일링...")
        print(f"   배치 크기: {config['batch_size']}")
        print(f"   Workers: {config['num_workers']}")
        print(f"   Pin Memory: {config['pin_memory']}")

        try:
            # DataLoader 생성
            train_loader, test_loader = create_unified_bid_dataloaders(
                db_engine=engine,
                schema=schema,
                batch_size=config["batch_size"],
                test_split=config["test_split"],
                shuffle_seed=42,
                num_workers=config["num_workers"],
                pin_memory=config["pin_memory"],
                streaming=config["streaming"],
                load_all_features=config["load_all_features"],
                chunk_size=1000000,
                feature_chunksize=1000,
                use_preprocessor=config["use_preprocessor"],
                test_mode=True,
                pair_limit=config["pair_limit"]
            )

            print(f"   Train 배치: {len(train_loader)}")

            # 프로파일링 실행
            output_dir = f"profiling_results/config_{i+1}_{config['name'].replace(' ', '_').replace('(', '').replace(')', '')}"
            profiler = CPUBottleneckProfiler(output_dir=output_dir)

            # 전체 파이프라인 프로파일링
            profile_dataloader_performance(train_loader, profiler, num_batches=20)

            # Collate 함수 단독 프로파일링 (첫 번째 설정에서만)
            if i == 0:
                print("   🔧 Collate 함수 단독 분석...")
                profile_collate_function_separately(
                    train_loader.dataset,
                    batch_size=config["batch_size"]
                )

        except Exception as e:
            print(f"   ❌ 오류 발생: {e}")
            continue

    # 4. 종합 분석 및 권장사항
    print("\n5️⃣ 종합 분석 및 권장사항...")
    generate_optimization_recommendations()

    print("\n🎯 프로파일링 완료!")
    print("📊 결과 확인:")
    print("   - profiling_results/config_1_현재_설정_실제_배치_크기/")
    print("   - profiling_results/config_2_GPU_최적화_설정/")
    print("   - profiling_results/config_3_멀티프로세싱_설정/")
    print("   - profiling_results/collate_only/")
    print("\n📈 TensorBoard 실행:")
    print("   tensorboard --logdir profiling_results")


def generate_optimization_recommendations():
    """
    CPU 병목 해결을 위한 최적화 권장사항 생성
    """

    recommendations = [
        "=== CPU 병목 해결 권장사항 ===\n",

        "🎯 즉시 적용 가능한 최적화:",
        "1. 배치 크기 증가: 256 → 512 (GPU 메모리 허용 시)",
        "   - 배치당 오버헤드 감소, GPU 활용도 향상",

        "2. GPU에서 직접 KJT 생성:",
        "   - collate_fn에서 .cuda() 호출하여 GPU에서 직접 생성",
        "   - CPU → GPU 전송 시간 단축",

        "3. 비동기 데이터 로딩:",
        "   - num_workers=2~4, pin_memory=True",
        "   - 단, DB 연결 이슈 주의",

        "\n🔧 중기 최적화 (구현 필요):",
        "4. 배치 사전 생성 (Batch Prefetching):",
        "   - 다음 배치를 미리 준비하여 GPU 대기 시간 최소화",

        "5. 메모리 사전 할당:",
        "   - KJT, Dense 텐서 메모리 풀링",
        "   - 배치마다 메모리 할당/해제 오버헤드 제거",

        "6. Database 최적화:",
        "   - 커넥션 풀링, 쿼리 캐싱",
        "   - 청크 크기 최적화",

        "\n🚀 고급 최적화 (장기):",
        "7. NVIDIA DALI 적용:",
        "   - GPU 기반 데이터 전처리 파이프라인",
        "   - CPU 병목 완전 제거",

        "8. 분산 처리:",
        "   - DataParallel/DistributedDataParallel",
        "   - 다중 GPU 활용",

        "\n📊 성능 목표:",
        "- 현재: GPU 사용률 40%",
        "- 목표: GPU 사용률 80%+",
        "- 예상 학습 속도 향상: 2x~3x"
    ]

    # 권장사항 파일로 저장
    output_path = Path("profiling_results/optimization_recommendations.txt")
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w") as f:
        f.write("\n".join(recommendations))

    # 콘솔에도 출력
    print("\n".join(recommendations))


def quick_profile():
    """
    빠른 프로파일링 (디버깅용)
    """
    print("⚡ 빠른 CPU 병목 검출...")

    # 기본 설정으로 간단히 실행
    db = DatabaseConnector()
    engine = db.engine

    # 스키마 설정 (.env에서 자동으로 설정 읽음)
    schema_config = {
        "pair_notice_id_cols": ["bidntceno", "bidntceord"],
        "pair_company_id_cols": ["bizno"],
        "metadata_path": "meta/metadata.csv"
    }
    schema = build_torchrec_schema_from_meta(**schema_config)

    # 작은 데이터셋으로 빠른 테스트
    train_loader, _ = create_unified_bid_dataloaders(
        db_engine=engine,
        schema=schema,
        batch_size=128,
        test_split=0.2,
        shuffle_seed=42,
        num_workers=0,
        pin_memory=False,
        streaming=True,
        load_all_features=True,
        use_preprocessor=True,
        test_mode=True,
        pair_limit=5000  # 매우 작은 데이터셋
    )

    # 빠른 프로파일링
    profiler = CPUBottleneckProfiler(output_dir="profiling_results/quick_test")
    profile_dataloader_performance(train_loader, profiler, num_batches=5)

    print("✅ 빠른 프로파일링 완료!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CPU 병목 지점 검출")
    parser.add_argument("--quick", action="store_true",
                       help="빠른 프로파일링 (작은 데이터셋)")

    args = parser.parse_args()

    if args.quick:
        quick_profile()
    else:
        main()