#!/usr/bin/env python3
"""
TFRecord 파일 내용을 확인하는 유틸리티 스크립트
"""

import tensorflow as tf
import json
from typing import Dict, Any, Optional
import os



def parse_tfrecord_example(example_proto):
    """TFRecord Example을 파싱하여 딕셔너리로 변환"""
    # Feature 정의 (동적으로 처리하기 위해 비워둠)
    feature_description = {}
    
    # Example 파싱
    example = tf.train.Example()
    example.ParseFromString(example_proto.numpy())
    
    result = {}
    for feature_name, feature in example.features.feature.items():
        # Feature 타입에 따라 값 추출
        if feature.HasField('int64_list'):
            values = list(feature.int64_list.value)
            result[feature_name] = values[0] if len(values) == 1 else values
        elif feature.HasField('float_list'):
            values = list(feature.float_list.value)
            result[feature_name] = values[0] if len(values) == 1 else values
        elif feature.HasField('bytes_list'):
            values = [v.decode('utf-8', errors='ignore') for v in feature.bytes_list.value]
            result[feature_name] = values[0] if len(values) == 1 else values
    
    return result

def inspect_tfrecord(file_path: str, max_records: int = 5, show_schema: bool = True):
    """
    TFRecord 파일의 내용을 검사
    
    Args:
        file_path: TFRecord 파일 경로
        max_records: 출력할 최대 레코드 수
        show_schema: 스키마 정보 출력 여부
    """
    print(f"🔍 TFRecord 파일 검사: {file_path}")
    print("=" * 80)
    
    # 압축 여부 확인
    compression = "GZIP" if file_path.endswith('.gz') else ""
    
    try:
        dataset = tf.data.TFRecordDataset(file_path, compression_type=compression)
        
        # 스키마 정보 수집
        all_features = set()
        feature_types = {}
        total_count = 0
        
        records = []
        for raw_record in dataset.take(max_records + 100):  # 스키마 파악을 위해 더 많이 읽기
            try:
                parsed = parse_tfrecord_example(raw_record)
                total_count += 1
                
                # 처음 max_records개만 저장
                if len(records) < max_records:
                    records.append(parsed)
                
                # 스키마 정보 수집
                for key, value in parsed.items():
                    all_features.add(key)
                    if key not in feature_types:
                        feature_types[key] = type(value).__name__
                        
            except Exception as e:
                print(f"❌ 레코드 파싱 실패: {e}")
                continue
        
        # 전체 레코드 수 계산 (더 정확하게)
        if total_count >= max_records + 100:
            total_count = sum(1 for _ in dataset)
        
        print(f"📊 전체 레코드 수: {total_count:,}")
        print(f"📋 총 {len(all_features)}개 피처")
        
        if show_schema:
            print("\n🗂️  스키마 정보:")
            print("-" * 40)
            for feature in sorted(all_features):
                feature_type = feature_types.get(feature, 'unknown')
                print(f"  {feature:<30} {feature_type}")
        
        print(f"\n📄 레코드 샘플 ({min(max_records, len(records))}개):")
        print("=" * 80)
        
        for i, record in enumerate(records):
            print(f"\n📝 레코드 #{i+1}:")
            print("-" * 40)
            for key, value in record.items():
                # 값이 너무 길면 자르기
                if isinstance(value, str) and len(value) > 100:
                    display_value = value[:100] + "..."
                elif isinstance(value, list) and len(value) > 10:
                    display_value = f"{value[:5]} ... (+{len(value)-5} more)"
                else:
                    display_value = value
                
                print(f"  {key:<25} : {display_value}")
        
    except Exception as e:
        print(f"❌ 파일 읽기 실패: {e}")

def search_records(file_path: str, search_key: str, search_value: Any, max_results: int = 10):
    """
    특정 조건에 맞는 레코드 검색
    
    Args:
        file_path: TFRecord 파일 경로
        search_key: 검색할 키
        search_value: 검색할 값
        max_results: 최대 결과 수
    """
    print(f"🔎 검색 조건: {search_key} = {search_value}")
    print("=" * 80)
    
    compression = "GZIP" if file_path.endswith('.gz') else ""
    dataset = tf.data.TFRecordDataset(file_path, compression_type=compression)
    
    found_count = 0
    total_checked = 0
    
    for raw_record in dataset:
        try:
            parsed = parse_tfrecord_example(raw_record)
            total_checked += 1
            
            if search_key in parsed and parsed[search_key] == search_value:
                found_count += 1
                print(f"\n✅ 발견된 레코드 #{found_count} (전체 {total_checked}번째):")
                print("-" * 40)
                for key, value in parsed.items():
                    if isinstance(value, str) and len(value) > 100:
                        display_value = value[:100] + "..."
                    else:
                        display_value = value
                    print(f"  {key:<25} : {display_value}")
                
                if found_count >= max_results:
                    break
                    
        except Exception as e:
            continue
    
    print(f"\n📊 검색 결과: {found_count}개 발견 (총 {total_checked}개 확인)")
    

def count_tfrecords(pattern, compression=None):
    files = tf.io.gfile.glob(pattern) if "*" in pattern else [pattern]
    ds = tf.data.TFRecordDataset(
        files,
        compression_type=compression,           # "GZIP" 사용 중이면 꼭 지정
        num_parallel_reads=tf.data.AUTOTUNE,
    )
    return ds.reduce(tf.constant(0, tf.int64), lambda x, _: x + 1).numpy()


if __name__ == "__main__":
    print("notices:",  count_tfrecords("output/tfrecord/notice_preprocessed.tfrecord.gz", compression="GZIP"))
    print("companies:",count_tfrecords("output/tfrecord/company_preprocessed.tfrecord.gz", compression="GZIP"))
    print("pairs:",    count_tfrecords("output/tfrecord/pairs.tfrecord.gz", compression="GZIP"))
    