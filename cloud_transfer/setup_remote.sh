#!/bin/bash
# RunPod 서버에서 프로젝트 셋업

mkdir -p /data/dev
cd /data/dev
git clone https://github.com/zoongahn/jodalroB-twoTower
cd jodalroB-twoTower
pip install -r requirements.txt

echo "Setup complete!"
