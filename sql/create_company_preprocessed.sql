-- company_preprocessed 테이블 생성
-- 전처리된 업체 데이터 (수치형, 범주형)

CREATE TABLE IF NOT EXISTS tmp.company_preprocessed (
    -- Primary Key
    bizno TEXT NOT NULL,

-- 수치형 컬럼 (원본)
emplyenum DOUBLE PRECISION,

-- 수치형 컬럼 (NULL 여부 플래그)
emplyenum_is_null DOUBLE PRECISION,

-- 범주형 컬럼 (인코딩된 값)
rgncd BIGINT,
rgnnm BIGINT,
cntrynm BIGINT,
mnfctdivnm BIGINT,
corpbsnsdivcd BIGINT,
hdoffcedivnm BIGINT,

-- 범주형 컬럼 (NULL 여부 플래그)
rgncd_is_null DOUBLE PRECISION,
rgnnm_is_null DOUBLE PRECISION,
cntrynm_is_null DOUBLE PRECISION,
mnfctdivnm_is_null DOUBLE PRECISION,
corpbsnsdivcd_is_null DOUBLE PRECISION,
hdoffcedivnm_is_null DOUBLE PRECISION,

-- Primary Key 제약조건
PRIMARY KEY (bizno) );

-- 코멘트
COMMENT ON TABLE step1.company_preprocessed IS '전처리된 업체 데이터 (수치형, 범주형)';

COMMENT ON COLUMN step1.company_preprocessed.bizno IS '사업자번호 (Primary Key)';

COMMENT ON COLUMN step1.company_preprocessed.emplyenum IS '종업원 수';