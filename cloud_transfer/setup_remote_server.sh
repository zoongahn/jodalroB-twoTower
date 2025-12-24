mkdir -p /data
mkdir -p /data/dev
cd /data
cd dev
git clone https://github.com/zoongahn/jodalroB-twoTower
cd jodalroB-twoTower

pip install requirements.txt

mkdir -p data/parquet

echo "Setup complete!"
