-- bid_two_tower 테이블 생성
-- 공고-업체 매핑 테이블 (Two-Tower 모델 학습용)

CREATE TABLE IF NOT EXISTS tmp.bid_two_tower (
    id BIGSERIAL,
    bidntceno TEXT,
    bidntceord TEXT,
    bizno TEXT
);

-- 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_bid_two_tower_id ON step1.bid_two_tower (id);

-- 코멘트
COMMENT ON TABLE step1.bid_two_tower IS '공고-업체 매핑 테이블 (Two-Tower 모델 학습용)';

COMMENT ON COLUMN step1.bid_two_tower.id IS '자동 증가 ID';

COMMENT ON COLUMN step1.bid_two_tower.bidntceno IS '공고번호';

COMMENT ON COLUMN step1.bid_two_tower.bidntceord IS '공고차수';

COMMENT ON COLUMN step1.bid_two_tower.bizno IS '사업자번호';