---
description: 2024.12.21.
---

# 그래서 제일 좋은 Chunking이 뭔데?

<figure><img src="../.gitbook/assets/dddd (1).png" alt=""><figcaption><p>그래서 제일 좋은 chunking이 뭐야?</p></figcaption></figure>

앞선 Chunking에 관한 2개 포스트([당신.. chunking이 뭔지 정확히 알아?](..-chunking-...md), [그래서 제일 좋은 chunking이 뭔데?](chunking.md))에 이어서 이번포스팅에는 가장 성능이 좋은 Chunking 전략이 어떤  방법인지 너무나도 궁금했던 필자가 진행한 실험내용과 고민들을 같이 보도록 하자.&#x20;

## Ⅰ. Background





## Ⅱ. Experiments







## Ⅲ. Conclusion

&#x20;실험결과를 보자.

<figure><img src="../.gitbook/assets/image (1).png" alt=""><figcaption><p>[ Fig. RC-Score from each chunking methodology] </p></figcaption></figure>



## Ⅳ. Limitation

* 특정 도메인 내용의 데이터를 사용하여 General한 성능으로 볼 수 없다.
* Retrieval Metric의 경우 아래와 같은 외부요인에 영향을 받을 수 밖에 없다.
  * Embedding Model의 성능
  * Evaluation Dataset의 구성 (품질)
* Ranking Combination Score의 계산시, 사전에 작성한 평가기준에 따라 각기다른 가중치값이 부여되고 이에 따라 전체 Score 값이 달라진다.







