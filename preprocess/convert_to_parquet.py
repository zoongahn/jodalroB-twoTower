import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
import logging
from typing import Dict, List, Optional, Union
from tqdm import tqdm
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from data.database_connector import DatabaseConnector


class DBToParquetConverter:
    """DB 테이블을 Parquet 파일로 변환하는 클래스"""
    
    def __init__(self, db_config: Optional[Dict] = None, output_dir: str = "data/parquet"):
        """
        Args:
            db_config: DB 연결 정보 (None이면 환경변수 사용)
            output_dir: Parquet 파일이 저장될 디렉토리
        """
        self.output_dir = output_dir
        self.db_connector = DatabaseConnector(db_config)
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
    def get_table_row_count(self, table_name: str) -> int:
        """테이블의 총 행 수 조회"""
        count_sql = f"SELECT COUNT(*) FROM {table_name}"
        # DatabaseConnector의 execute_query 사용 (text() 처리됨)
        result_df = self.db_connector.execute_query(count_sql)
        row_count = int(result_df.iloc[0, 0])
        logger.info(f"테이블 {table_name}: {row_count:,} 행")
        return row_count
    
    def convert_table_to_parquet(
        self,
        table_name: str,
        chunk_size: Optional[int] = None,
        compression: str = 'snappy',
        use_chunked_parquet: bool = True,
        size_threshold: int = 1000000
    ) -> str:
        """
        단일 테이블을 Parquet으로 변환
        
        Args:
            table_name: 변환할 테이블 이름
            chunk_size: 메모리 효율성을 위한 청크 크기 (None이면 자동 결정)
            compression: 압축 방식 ('snappy', 'gzip', 'brotli', 'lz4')
            use_chunked_parquet: 청크별로 별도 파일로 저장할지 여부
            size_threshold: 청크 처리 여부를 결정하는 행 수 임계값
            
        Returns:
            str: 생성된 파일 경로
        """
        logger.info(f"테이블 변환 시작: {table_name}")
        
        # 총 행 수 조회
        total_rows = self.get_table_row_count(table_name)
        
        # 청크 사이즈 자동 결정
        if chunk_size is None:
            if total_rows <= size_threshold:
                # 작은 테이블: 청크 없이 한번에 읽기
                return self._convert_without_chunking(table_name, compression)
            else:
                # 큰 테이블: 기본 청크 사이즈 사용
                chunk_size = 100000
        
        total_chunks = math.ceil(total_rows / chunk_size)
        output_file = os.path.join(self.output_dir, f"{table_name}.parquet")
        
        if use_chunked_parquet and total_chunks > 1:
            # 대용량 데이터: 청크별 파일로 저장
            return self._convert_large_table(table_name, chunk_size, compression, total_rows)
        else:
            # 소용량 데이터: 단일 파일로 저장
            return self._convert_small_table(table_name, chunk_size, compression, output_file)
    
    def _convert_without_chunking(self, table_name: str, compression: str) -> str:
        """작은 테이블을 청크 없이 한번에 읽어서 변환"""
        output_file = os.path.join(self.output_dir, f"{table_name}.parquet")
        query = f"SELECT * FROM {table_name}"
        
        logger.info(f"한번에 읽기 (청크 없음): {table_name}")
        
        # 전체 데이터를 한번에 읽기
        df = self.db_connector.execute_query(query)
        
        if len(df) > 0:
            # Parquet 저장
            df.to_parquet(
                output_file,
                compression=compression,
                index=False,
                engine='pyarrow'
            )
            
            logger.info(f"저장 완료: {output_file} ({len(df):,} 행)")
            return output_file
        else:
            logger.warning(f"테이블 {table_name}에 데이터가 없습니다.")
            return None
    
    def _convert_small_table(self, table_name: str, chunk_size: int, compression: str, output_file: str) -> str:
        """소용량 테이블을 단일 Parquet 파일로 변환"""
        query = f"SELECT * FROM {table_name}"
        
        # 청크 단위로 읽어서 하나의 파일로 저장 - DatabaseConnector의 청크 메서드 사용
        chunks = []
        for chunk in tqdm(self.db_connector.iter_sql_chunks(query, chunksize=chunk_size), desc=f"Loading {table_name}"):
            chunks.append(chunk)
        
        # 모든 청크를 하나로 합치기
        if chunks:
            df = pd.concat(chunks, ignore_index=True)
            
            # Parquet 저장
            df.to_parquet(
                output_file,
                compression=compression,
                index=False,
                engine='pyarrow'
            )
            
            logger.info(f"저장 완료: {output_file} ({len(df):,} 행)")
            return output_file
        else:
            logger.warning(f"테이블 {table_name}에 데이터가 없습니다.")
            return None
    
    def _convert_large_table(self, table_name: str, chunk_size: int, compression: str, total_rows: int) -> str:
        """대용량 테이블을 청크별 Parquet 파일로 변환"""
        output_dir = os.path.join(self.output_dir, table_name)
        os.makedirs(output_dir, exist_ok=True)
        
        query = f"SELECT * FROM {table_name}"
        total_chunks = math.ceil(total_rows / chunk_size)
        
        # 청크별로 처리 - DatabaseConnector의 청크 메서드 사용
        chunk_files = []
        with tqdm(total=total_chunks, desc=f"Converting {table_name}") as pbar:
            for i, chunk in enumerate(self.db_connector.iter_sql_chunks(query, chunksize=chunk_size)):
                chunk_file = os.path.join(output_dir, f"chunk_{i:04d}.parquet")
                
                chunk.to_parquet(
                    chunk_file,
                    compression=compression,
                    index=False,
                    engine='pyarrow'
                )
                
                chunk_files.append(chunk_file)
                pbar.update(1)
        
        logger.info(f"청크 파일 생성 완료: {len(chunk_files)}개 파일")
        
        # 메타데이터 파일 생성
        metadata_file = os.path.join(output_dir, "metadata.yaml")
        metadata = {
            'table_name': table_name,
            'total_rows': total_rows,
            'chunk_size': chunk_size,
            'total_chunks': len(chunk_files),
            'compression': compression,
            'files': [os.path.basename(f) for f in chunk_files]
        }
        
        with open(metadata_file, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)
            
        return output_dir
    
    def convert_multiple_tables(
        self,
        table_configs: List[Dict],
        max_workers: int = 4
    ) -> Dict[str, str]:
        """
        여러 테이블을 병렬로 변환
        
        Args:
            table_configs: [{'name': 'table1', 'chunk_size': 100000, 'compression': 'snappy'}, ...]
            max_workers: 병렬 처리 스레드 수
            
        Returns:
            Dict[str, str]: {table_name: output_path}
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 작업 제출
            future_to_table = {
                executor.submit(
                    self.convert_table_to_parquet,
                    config['name'],
                    config.get('chunk_size', None),  # None이면 자동 결정
                    config.get('compression', 'snappy'),
                    config.get('use_chunked_parquet', True)
                ): config['name']
                for config in table_configs
            }
            
            # 결과 수집
            for future in as_completed(future_to_table):
                table_name = future_to_table[future]
                try:
                    output_path = future.result()
                    results[table_name] = output_path
                    logger.info(f"테이블 {table_name} 변환 완료")
                except Exception as exc:
                    logger.error(f"테이블 {table_name} 변환 실패: {exc}")
                    results[table_name] = None
        
        return results
    
    def load_parquet_table(self, table_name: str) -> pd.DataFrame:
        """
        변환된 Parquet 파일을 다시 로드
        
        Args:
            table_name: 로드할 테이블 이름
            
        Returns:
            pd.DataFrame: 로드된 데이터
        """
        single_file = os.path.join(self.output_dir, f"{table_name}.parquet")
        chunked_dir = os.path.join(self.output_dir, table_name)
        
        if os.path.exists(single_file):
            # 단일 파일인 경우
            return pd.read_parquet(single_file)
        elif os.path.exists(chunked_dir):
            # 청크 파일인 경우
            parquet_files = [
                os.path.join(chunked_dir, f) 
                for f in os.listdir(chunked_dir) 
                if f.endswith('.parquet')
            ]
            
            if parquet_files:
                return pd.read_parquet(parquet_files)
            else:
                raise FileNotFoundError(f"청크 파일을 찾을 수 없습니다: {chunked_dir}")
        else:
            raise FileNotFoundError(f"Parquet 파일을 찾을 수 없습니다: {table_name}")


def main():
    """사용 예시"""
    
    # DatabaseConnector는 환경변수를 자동으로 읽음
    # .env 파일에 다음 설정이 있어야 함:
    # POSTGRES_HOST=localhost
    # POSTGRES_PORT=5432
    # POSTGRES_DB=bidding_db
    # POSTGRES_USER=your_username
    # POSTGRES_PASSWORD=your_password
    
    # 또는 직접 설정 가능
    # db_config = {
    #     'host': 'localhost',
    #     'port': 5432,
    #     'database': 'bidding_db',
    #     'user': 'your_username',
    #     'password': 'your_password'
    # }
    
    # 변환기 생성 (환경변수 사용)
    converter = DBToParquetConverter(db_config=None, output_dir="parquet")
    
    # Two-Tower 모델용 테이블 설정
    table_configs = [
        {
            'name': 'notice_preprocessed',  # 실제 테이블명
            'chunk_size': 50000,
            'compression': 'snappy',
            'use_chunked_parquet': True
        },
        {
            'name': 'company_preprocessed',  # 실제 테이블명
            'chunk_size': None,  # 자동 결정 (작은 테이블이면 청크 없이)
            'compression': 'snappy',
            'use_chunked_parquet': False  # 단일 파일로 저장
        }
    ]
    
    # 병렬 변환 실행
    results = converter.convert_multiple_tables(table_configs, max_workers=3)
    
    # 결과 출력
    for table_name, output_path in results.items():
        if output_path:
            print(f"✅ {table_name}: {output_path}")
        else:
            print(f"❌ {table_name}: 변환 실패")
    
    # 사용 예시: 변환된 파일 로드
    try:
        notices_df = converter.load_parquet_table('notice_preprocessed')
        print(f"공고 데이터 로드: {len(notices_df):,} 행")
    except Exception as e:
        print(f"로드 실패: {e}")
    
    # 연결 종료
    converter.db_connector.close()


if __name__ == "__main__":
    main()