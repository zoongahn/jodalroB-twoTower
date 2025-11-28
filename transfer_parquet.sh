#!/bin/bash
# Parquet 파일을 RunPod 서버 1로 전송

SSH_KEY="~/.ssh/id_ed25519"
HOST="216.81.245.26"
PORT="43567"
USER="root"
REMOTE_PATH="/data/dev/jodalroB-twoTower/data/parquet"

# 원격 디렉토리 생성
ssh -i $SSH_KEY -p $PORT -o StrictHostKeyChecking=no ${USER}@${HOST} "mkdir -p $REMOTE_PATH"

# 파일 전송
scp -i $SSH_KEY -P $PORT -o StrictHostKeyChecking=no \
    data/parquet/notice.parquet \
    data/parquet/pairs.parquet \
    data/parquet/company.parquet \
    ${USER}@${HOST}:${REMOTE_PATH}/

echo "Transfer complete!"
