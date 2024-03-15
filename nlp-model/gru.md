---
description: GRU(Gated Recurrent Unit)에 대해 알아봅시다.
---

# 👾 GRU

## Ⅰ. GRU (Gated Recurrent Unit)

GRU(Gated Recurrent Unit)은 2014년 뉴욕대학교 조경현 교수님이 집필한 논문에서 제안되었습니다. LSTM의 장기 의존성 문제에 대한 해결책을 유지하면서 은닉상태를 업데이트하는 계산을 줄였고 성능은 LSTM과 유사하면서 복잡했던 LSTM의 구조를 간단화 시킨 모델입니다.

기존의 LSTM에서는 출력, 입력, 삭제 게이트라는 3개의 게이트가 존재했다면 GRU에서는 업데이트 게이트과 리셋 게이트 3가지 게이트만 존재합니다. GRU는 LSTM보다 학습 속도가 빠르다고 알려져있지만 여러 평가에서 GRU는 LSTM과 비슷한 성능을 보입니다.

<figure><img src="../.gitbook/assets/image (26).png" alt="" width="563"><figcaption></figcaption></figure>

$$
r_t = \sigma(W_{xr}x_t+W_{hr}h_{t-1}+b_r)\\z_t=\sigma(W_{xz}x_t+W_{hz}h_{t-1}+b_z)\\g_t=tanh(W_{hg}(r_t\otimes h_{t-1})+W_{xg}x_t+b_g)\\h_t=(1-z_t)\otimes g_t + z_t \otimes h_{t-1}
$$

GRU와 LSTM중 어떤 것이 모델의 성능면에서 더 낫다라고 말할 수는 없으나 기존의 LSTM 모델을 사용하더라도 최적의 하이퍼파라미터를 찾아냈다면 굳이 GRU로 바꿀 필요는 없을것입니다. 그러나 경험적으로 데이터 양이 적을 때 매개 변수의 양이 적은 GRU가 더 낫고, 데이터 양이 많으면 LSTM이 더 나은 경향이 있는것 같습니다.

## Implementation

GRU 모델을 이용한 IMDB 리뷰 분류를 구현해보겠습니다.

{% code lineNumbers="true" %}
```python




```
{% endcode %}
