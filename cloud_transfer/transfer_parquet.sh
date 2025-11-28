#!/bin/bash
# Parquet 파일을 RunPod 서버 1로 전송 (rsync 압축 전송)

SSH_KEY="~/.ssh/id_ed25519"
HOST="216.81.245.26"
PORT="43567"
USER="root"
REMOTE_PATH="/work/jodalroB-twoTower/data/parquet"

# 원격 디렉토리 생성
ssh -i $SSH_KEY -p $PORT -o StrictHostKeyChecking=no ${USER}@${HOST} "mkdir -p $REMOTE_PATH"

# rsync로 압축 전송
rsync -avz --progress -e "ssh -i $SSH_KEY -p $PORT -o StrictHostKeyChecking=no" \
    data/parquet/*.parquet \
    ${USER}@${HOST}:${REMOTE_PATH}/

echo "Transfer complete!"
