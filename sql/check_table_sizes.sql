-- 데이터베이스의 모든 테이블 용량 조사
-- PostgreSQL용

-- 1. 전체 테이블 크기 요약 (크기 순 정렬)
SELECT
    schemaname AS schema_name,
    tablename AS table_name,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS total_size,
    pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) AS table_size,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename) - pg_relation_size(schemaname||'.'||tablename)) AS indexes_size,
    pg_total_relation_size(schemaname||'.'||tablename) AS total_bytes
FROM pg_tables
WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
ORDER BY total_bytes DESC;


-- 2. 특정 스키마의 테이블 상세 정보 (행 수 포함)
SELECT
    schemaname AS schema_name,
    tablename AS table_name,
    n_tup_ins AS rows_inserted,
    n_tup_upd AS rows_updated,
    n_tup_del AS rows_deleted,
    n_live_tup AS live_rows,
    n_dead_tup AS dead_rows,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS total_size,
    pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) AS table_size,
    pg_size_pretty(pg_indexes_size(schemaname||'.'||tablename)) AS indexes_size,
    last_vacuum,
    last_autovacuum,
    last_analyze,
    last_autoanalyze
FROM pg_stat_user_tables
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;


-- 3. 스키마별 총 용량
SELECT
    schemaname AS schema_name,
    COUNT(*) AS table_count,
    pg_size_pretty(SUM(pg_total_relation_size(schemaname||'.'||tablename))) AS total_size,
    SUM(pg_total_relation_size(schemaname||'.'||tablename)) AS total_bytes
FROM pg_tables
WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
GROUP BY schemaname
ORDER BY total_bytes DESC;


-- 4. 프로젝트 관련 테이블만 (step1, step2 등)
SELECT
    schemaname AS schema_name,
    tablename AS table_name,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS total_size,
    pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) AS table_size,
    pg_size_pretty(pg_indexes_size(schemaname||'.'||tablename)) AS indexes_size,
    (SELECT COUNT(*) FROM information_schema.columns
     WHERE table_schema = schemaname AND table_name = tablename) AS column_count
FROM pg_tables
WHERE schemaname IN ('step1', 'step2', 'step3', 'public')
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;


-- 5. Notice, Company, Pair 테이블 상세 분석
WITH table_info AS (
    SELECT
        schemaname AS schema_name,
        tablename AS table_name,
        pg_total_relation_size(schemaname||'.'||tablename) AS total_bytes,
        pg_relation_size(schemaname||'.'||tablename) AS table_bytes,
        pg_indexes_size(schemaname||'.'||tablename) AS indexes_bytes
    FROM pg_tables
    WHERE tablename LIKE '%notice%'
       OR tablename LIKE '%company%'
       OR tablename LIKE '%pair%'
)
SELECT
    schema_name,
    table_name,
    pg_size_pretty(total_bytes) AS total_size,
    pg_size_pretty(table_bytes) AS table_size,
    pg_size_pretty(indexes_bytes) AS indexes_size,
    ROUND(100.0 * indexes_bytes / NULLIF(total_bytes, 0), 2) AS index_ratio_pct
FROM table_info
ORDER BY total_bytes DESC;


-- 6. 메모리 추정 (행 수 기반)
-- Notice 테이블 예시
SELECT
    'step1.notice' AS table_name,
    COUNT(*) AS row_count,
    pg_size_pretty(pg_total_relation_size('step1.notice')) AS current_disk_size,
    -- 예상 메모리 사용량 (압축 없이)
    pg_size_pretty(COUNT(*) * (
        -- 범주형 8 columns * 8 bytes (long)
        8 * 8 +
        -- 수치형 예상 (metadata에서 확인 필요)
        10 * 8 +
        -- 텍스트 임베딩 (768 * 4 bytes per float)
        768 * 4
    )::bigint) AS estimated_memory_fp32,
    pg_size_pretty(COUNT(*) * (
        -- FP16 사용 시
        8 * 8 + 10 * 4 + 768 * 2
    )::bigint) AS estimated_memory_fp16
FROM step1.notice;


-- 7. 데이터베이스 전체 크기
SELECT
    pg_database.datname AS database_name,
    pg_size_pretty(pg_database_size(pg_database.datname)) AS database_size,
    pg_database_size(pg_database.datname) AS size_bytes
FROM pg_database
WHERE datname = current_database();
