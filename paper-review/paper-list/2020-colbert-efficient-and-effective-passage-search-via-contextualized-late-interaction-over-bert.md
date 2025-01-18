---
description: 2024.11.22.
---

# \[2020] ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT

<figure><img src="../../.gitbook/assets/colBERT paper_cover.png" alt=""><figcaption><p>ColBERT : Efficient and Effective Passage Search via Contextualized Late Interaction over BERT</p></figcaption></figure>

> **Paper Link** : [https://arxiv.org/pdf/2004.12832](https://arxiv.org/pdf/2004.12832)

```
※ 본 포스팅은 논문의 가장 중요한 내용에 대한 리뷰를 정리하여 올리기 때문에 다소 축약되거나 의역된 내용이 많습니
```

## 1. Background

### 1. Neural Ranking Model

<figure><img src="../../.gitbook/assets/image (37).png" alt=""><figcaption></figcaption></figure>

### 2. Scehematic diagram illustrating query-document matching paradigm in neural IR

<figure><img src="../../.gitbook/assets/image (38).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (39).png" alt=""><figcaption></figcaption></figure>

## 2. ColBERT

### 1. ColBERT Contribution

<figure><img src="../../.gitbook/assets/image (40).png" alt=""><figcaption></figcaption></figure>

### 2. ColBERT Architecture : Query Encoder

<figure><img src="../../.gitbook/assets/image (42).png" alt=""><figcaption></figcaption></figure>

### 3. ColBERT Architecture : Document Encoder

<figure><img src="../../.gitbook/assets/image (43).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (44).png" alt=""><figcaption></figcaption></figure>

### 4. ColBERT Architecture : Late Interaction

<figure><img src="../../.gitbook/assets/image (45).png" alt=""><figcaption></figcaption></figure>

## 3. Re-ranking

<figure><img src="../../.gitbook/assets/image (46).png" alt=""><figcaption></figcaption></figure>

## 4. e2e Retrieval

<figure><img src="../../.gitbook/assets/image (47).png" alt=""><figcaption></figcaption></figure>

<figure><img src="../../.gitbook/assets/image (48).png" alt=""><figcaption></figcaption></figure>

## 5. Experimental Evaluation

<figure><img src="../../.gitbook/assets/image (49).png" alt=""><figcaption></figcaption></figure>

