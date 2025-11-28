#!/usr/bin/env python3
"""학습 디버깅: NaN/Inf 원인 파악"""

import torch
import torch.nn.functional as F

# Contrastive loss 함수 복사
def contrastive_loss(notice_emb, company_emb, temperature=0.07):
    batch_size = notice_emb.size(0)

    # L2 정규화
    notice_emb = F.normalize(notice_emb, p=2, dim=1)
    company_emb = F.normalize(company_emb, p=2, dim=1)

    # 유사도 계산
    logits = torch.matmul(notice_emb, company_emb.T) / temperature

    # Label: diagonal은 positive pair
    labels = torch.arange(batch_size, device=notice_emb.device)

    # CrossEntropy Loss (양방향)
    loss_notice = F.cross_entropy(logits, labels)
    loss_company = F.cross_entropy(logits.T, labels)

    loss = (loss_notice + loss_company) / 2
    return loss

# 테스트 케이스
print("=== NaN/Inf 원인 테스트 ===\n")

# 1. 정상 케이스
print("1. 정상 케이스:")
notice = torch.randn(256, 128)
company = torch.randn(256, 128)
loss = contrastive_loss(notice, company, temperature=0.07)
print(f"   Loss: {loss.item():.4f}")

# 2. Zero embedding
print("\n2. Zero Embedding 케이스:")
notice_zero = torch.zeros(256, 128)
company_zero = torch.zeros(256, 128)
try:
    loss = contrastive_loss(notice_zero, company_zero, temperature=0.07)
    print(f"   Loss: {loss.item():.4f}")
except Exception as e:
    print(f"   Error: {e}")

# 3. 매우 큰 값
print("\n3. 매우 큰 값 케이스:")
notice_large = torch.randn(256, 128) * 1000
company_large = torch.randn(256, 128) * 1000
loss = contrastive_loss(notice_large, company_large, temperature=0.07)
print(f"   Loss: {loss.item():.4f}")

# 4. Temperature = 0
print("\n4. Temperature = 0 케이스:")
try:
    loss = contrastive_loss(notice, company, temperature=0.0)
    print(f"   Loss: {loss.item():.4f}")
except Exception as e:
    print(f"   Error: {e}")

# 5. 매우 작은 temperature
print("\n5. 매우 작은 Temperature 케이스:")
loss = contrastive_loss(notice, company, temperature=0.001)
print(f"   Loss: {loss.item():.4f}")

# 6. Normalize 전 통계 확인
print("\n6. Embedding 통계:")
print(f"   Notice - mean: {notice.mean():.4f}, std: {notice.std():.4f}, min: {notice.min():.4f}, max: {notice.max():.4f}")
print(f"   Company - mean: {company.mean():.4f}, std: {company.std():.4f}, min: {company.min():.4f}, max: {company.max():.4f}")

notice_norm = F.normalize(notice, p=2, dim=1)
print(f"\n   After normalize:")
print(f"   Notice - mean: {notice_norm.mean():.4f}, std: {notice_norm.std():.4f}")

# 7. Logits 범위 확인
print("\n7. Logits 분석:")
logits = torch.matmul(notice_norm, F.normalize(company, p=2, dim=1).T) / 0.07
print(f"   Logits - min: {logits.min():.4f}, max: {logits.max():.4f}, mean: {logits.mean():.4f}")
print(f"   Logits에 exp 적용 후 - min: {torch.exp(logits).min():.4f}, max: {torch.exp(logits).max():.4f}")
if torch.isinf(torch.exp(logits)).any():
    print(f"   ⚠️ Logits exp에서 Inf 발생!")

print("\n=== 분석 완료 ===")
