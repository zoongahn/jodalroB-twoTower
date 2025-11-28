
# -*- coding: cp949 -*-
import numpy as np

path = '/data/dev/jodalroB-twoTower/data/embeddings/notice.npz'
data = np.load(path, allow_pickle=True)
ids = data['ids']

print("ids.shape:", ids.shape)
print("ids dtype:", ids.dtype)
print("첫 5개:", ids[:5])
print("첫 원소 type:", type(ids[0]))
