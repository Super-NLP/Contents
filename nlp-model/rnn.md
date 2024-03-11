---
description: RNN(Recurrent Neural Network)에 대해서 알아봅시다.
---

# 👾 RNN

## Ⅰ. RNN (Recurrent Neural Network, RNN)

RNN(Recurrent Neural Network)은 시퀀스 모델로 입력과 출력을 시퀀스 단위로 처리하는 모델이면서 자연어처리에 가장 적합한 기본 모델중 하나입니다.

### 1. 순환 신경망 (Recurrent Neural Network, RNN)

은닉층에서 활성화함수를 지난 값은 오직 출력층 방향으로만 향하게 되고 이렇게 처리하는 방식을 피드 포워드 신경망(Feed Forwad Neural Network)라고 합니다. 다만, RNN은 은닉층의 노드에서 활성화 함수를 통해 나온 결과값을 출력층 방향으로도 보내면서, 다시 은닉층 노드의 다음 계산의 입력으로 보내는 특징을 가집니다.

<figure><img src="../.gitbook/assets/image (21).png" alt="" width="375"><figcaption></figcaption></figure>

$$x$$를 Input Layer의 입력벡터, $$y$$를 Output Layer의 출력벡터로 보게됩니다. 실제로는 편향 $$b$$도 입력으로 존재할 수 있지만 앞으로의 그림에선 생략하겠습니다. RNN에서 은닉층에서 활성화 함수를 통해 결과를 내보내는 역할을 하는 노드를 셀(cell)이라고 하는데 이 cell은 이전의 값을 기억하려고 하는 일종의 메모리 역할을 수행하므로 메모리 셀 혹은 RNN 셀이라고 표현합니다.

은닉층의 메모리 셀은 각각의 시점(time step)에서 바로 이전 시점에서의 은닉층의 메모리 셀에서 나온 값을 자신의 입력으로 사용하는 재귀적 활동을 하고있습니다. 현재 시점을 변수 $$t$$로 표현하면 현재 시점 $$t$$에서의 메모리 셀이 갖고있는 값은 과거의 메모리 셀들의 값에 영향을 받은 것임을 의미합니다.

메모리 셀이 Input layer 방향 또는 그 다음 시점 $$t+1$$의 자신에게 보내는 값을 은닉상태(hidden state)라고 합니다. 다시말해 $$t$$시점의 메모리 셀은 $$t-1$$시점의 메모리 셀이 보낸 은닉 상태값을 $$t$$시점의 은닉 상태 계산을 위한 입력값으로 사용합니다.

<figure><img src="../.gitbook/assets/image (22).png" alt="" width="375"><figcaption></figcaption></figure>

## Ⅱ. RNN의 원리

조금 더 자세하게 원리를 알아봅시다.

<figure><img src="../.gitbook/assets/image (23).png" alt="" width="188"><figcaption></figcaption></figure>

현재 시점 $$t$$에서의 은닉 상태값을 $$h_t$$라고 정의하겠습니다. 은닉층의 메모리 셀은 $$h_t$$를 계산하기 위해서 총 2개의 가중치를 가지게 됩니다. 하나는 Input layer에서 입력값을 위한 가중치 $$W_x$$이고, 하나는 이전 시점 $$t-1$$의 은닉 상태값인 $$h_{t-1}$$을 위한 가중치 $$W_h$$입니다. 이를 수식으로 표현하면 아래와 같이 나타낼 수 있습니다.

* **은닉층** :$$h_t = tanh(W_x x_t + W_h h_{t-1}+b)$$
* **출력층** :$$y_t = f(W_y h_t +b )$$

<mark style="color:blue;">단,</mark> $$f$$<mark style="color:blue;">는 비선형 활성화 함수중 하나입니다!!</mark>

RNN의 은닉층 연산을 벡터와 행렬 연산으로 이해할 수 있습니다. NLP에서 RNN의 입력 $$x_t$$는 대부분의 경우에서 단어 벡터로 간주할 수 있는데, 단어 벡터의 차원을 $$d$$라고 하고, 은닉 상태의 크기를 $$D_h$$라고 했을 때 각 벡터와 행렬의 크기는 아래와 같이 얘기할 수 있습니다.

* $$x_t : (d \times 1)$$
* $$W_x : (D_h \times d)$$
* $$W_h : (D_h \times D_h)$$
* $$h_{t-1} : (D_h \times 1)$$
* $$b : (D_h \times 1)$$

배치 크기가 1이고, $$d$$와 $$D_h$$ 두 값 모두를 4로 가정하였을 때, RNN의 은닉층 연산을 그림으로 표현하면 아래와 같습니다.

<figure><img src="../.gitbook/assets/image (24).png" alt=""><figcaption></figcaption></figure>

이때 $$h_t$$를 계산하기 위한 활성화 함수로 주로 $$tanh()$$가 사용되지만 경우에 따라 $$ReLU()$$를 사용하기도 합니다. 각각의 가중치 $$W_x, W_h, W_y$$의 값들은 모든 시점에서 동일하게 공유합니다. 그러나 만약 은닉층이 2개 이상일 경우 각 은닉층의 가중치는 서로 다릅니다. 하고자하는 task에 따라 활성화 함수를 적절하게 바꾸어가면서 사용하면 됩니다. 이진분류의 경우 $$sigmoid()$$를, 다중분류라면 $$softmax()$$를 사용하듯 말입니다.

## Ⅲ. RNN 구현

pytorch를 이용한 RNN모델을 구현해봅시다. 사실 RNN모델은 torch안에 내장되어 있어 큰 소요를 들이지 안히고 구현할 수 있습니다.

{% code lineNumbers="true" %}
```python
import torch
import torch.nn as nn

# hyperparameter
input_size = 5 # 입력의 크기
hidden_size = 8 # 은닉 상태의 크기

# (batch size, time step, input size)
inputs = torch.Tensor(1, 10, 5)

# batch_first = True : 입력 텐서의 첫번째 차원이 배치 크기이다!
cell = nn.RNN(input_size, hidden_size, batch_first=True)

# RNN Cell은 반환값이 2개
# 첫 번째 : 모든 시점(timesteps)의 은닉 상태
# 두 번째 : 마지막 시점(timestep)의 은닉 상태
outputs, _status = cell(inputs)
print(f'outputs : {outputs.shape} / _status : {_status.shape}')

# ----------------------------------------------------------
# Deep RNN

inputs2 = torch.Tensor(1, 10, 5)
cell2  = nn.RNN(input_size=5, hidden_size=8, num_layers=2, batch_first=True)
outputs2, _status2 = cell2(inputs2)
print(f'outputs : {outputs2.shape} / _status : {_status2.shape}')
```
{% endcode %}

## Ⅳ Bidirectional RNN

이러한 RNN구조도 잘 살펴보면 하나의 방향으로만 학습이 진행됨을 알 수 있습니다. 이를 개선한 것이 바로 양방향 순환 신경망(Bidirectional Recurrent Neural Net)이라고 하는데 $$t$$시점에서의 출력값을 예측할 때 이전 시점의 데이터 뿐 아니라 이후 데이터로도 예측할 수 있다는 아이디어입니다. 한가지 예시를 들어보겠습니다.

```
Exercise is very effective at [          ] belly fat.

1) reducing
2) increasing
3) multiplying
```

위 질문은 '운동은 복부 지방을 \[          ] 효과적이다.' 라는 문장이고 빈칸안에 들어갈 것은 reducing(줄이는 것)입니다.\
빈칸을 맞추는데 가장 큰 역할을 한 것은 바로 빈칸 다음에 나오는 belly fat(복부 지방) 입니다. 즉, 과거 시점뿐만 아니라 이후 시점에서도 힌트를 얻을 수 있다는 것이죠. 그렇게 고안된 것이 BiRNN입니다.

<figure><img src="../.gitbook/assets/image (25).png" alt=""><figcaption></figcaption></figure>

{% code lineNumbers="true" %}
```python
...

cell_bi  = nn.RNN(input_size=5, hidden_size=8, num_layers=2, batch_first=True, bidirectional=True)

...
```
{% endcode %}

구현은 기존 RNN 모델에 `bidirectional` 옵션만 True값을 주면 됩니다.
