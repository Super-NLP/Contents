---
description: LSTM에 대해서 알아봅시다.
---

# 👾 LSTM

## Ⅰ. 장단기 메모리(Long Short-Term Memory)

RNN의 다른 이름은 가장 단순한 형태의 RNN이라고 하여 ‘Vanilla RNN’이라고 부릅니다.

### 1. Vanila RNN의 한계

<figure><img src="../.gitbook/assets/image.png" alt="" width="375"><figcaption></figcaption></figure>

앞에서 바닐라 RNN은 출력 결과가 이전의 계산 결과에 의존한다는 것을 언급했습니다. 바닐라 RNN은 비교적 짧은 시퀀스(sequence)에 대해서만 효과를 보이는 단점이 있고 바닐라 RNN의 시점(time step)이 길어질 수록 앞의 정보가 뒤로 충분히 전달되지 못하는 현상이 발생합니다. 위의 그림은 첫 번째 입력값인 $$x_1$$의 정보량을 짙은 남색으로 표현했을 때, 색이 점차 얕아지는것으로 시점이 지날수록 $$x_1$$의 정보량이 손실되어가는 과정을 표현한 것입니다. 뒤로 갈수록 $$x_1$$의 정보량은 손실되고, 시점이 충분히 긴 상황에서는 $$x_1$$의 전체 정보에 대한 영향력이 거의 의미가 없다고 할 수 있습니다. 이를 쉽게 표현하면 RNN이 충분한 기억력을 가지고 있지 못한다고 이해할 수 있고 이를 **장기 의존성 문제 (the problem of Long-Term Dependencies)**라고 한다.

### 2. Vanila RNN 내부파악

<figure><img src="../.gitbook/assets/image (1).png" alt="" width="375"><figcaption></figcaption></figure>

LSTM에 대해서 이해해보기 전에 바닐라 RNN을 한번 더 알아봅시다. 위의 그림은 Vanilla RNN의 내부구조입니다. 편의성을 위해 위 그림에선 편향($$b$$)을 생략하여 표현하였습니다. 만일 편향 $$b$$를 그린다면 $$x_t$$옆에에 $$tanh()$$로 향하는 또 하나의 입력선을 그리면 된다.

$$
h_t = tanh(W_xx_t+W_hh_{t-1}+b)
$$

Vanilla RNN은 $$x_t$$와 $$h_{t-1}$$라는 2개의 입력이 각각의 가중치와 곱해져서 메모리 셀의 입력이 되고 이를 $$tanh()$$ 함수의 입력으로 사용하고 이 값은 은닉층의 출력인 은닉 상태가 됩니다. 이제 이러한 RNN에서 벗어나 LSTM을 알아보도록 하겠습니다.

## Ⅱ. LSTM (Long-Short Term Memory)

<figure><img src="../.gitbook/assets/image (2).png" alt="" width="563"><figcaption></figcaption></figure>

위 그림은 LSTM의 전체적인 내부 구성을 보여줍니다. Vanilla RNN의 이러한 단점을 보완한 모델을 장단기 메모리(Long Short-Term Memory)라고 하고, 줄여서 LSTM이라고 합니다. LSTM은 은닉층의 메모리 셀에 입력 게이트(input gate), 망각 게이트(forget gate), 출력 게이트(output gate)를 추가하여 불필요한 기억을 지우고, 기억해야할 것들을 정하게 됩니다. 즉, LSTM은 기존의 RNN보다 조금 더 복잡한 구성을 지니고 셀 상태(cell state)라는 값을 추가하였고 위의 그림에서는 t시점의 셀 상태를 $$C_t$$로 표현하고 있습니다. 그렇기 때문에 기존의 RNN과 비교하였을 때 긴 시퀀스의 입력을 처리하는데 탁월한 성능을 보입니다.

<figure><img src="../.gitbook/assets/image (3).png" alt="" width="563"><figcaption></figcaption></figure>

위 그림이 셀 상태(cell state)를 나타내는 부분이다. 왼쪽에서 오른쪽으로 가는 굵은 선을 말하며, 셀 상태(cell state) 또한 이전에 배운 은닉 상태처럼 이전 시점의 셀 상태가 다음 시점의 셀 상태를 구하기 위한 입력으로서 사용된다. 은닉 상태값과 셀 상태값을 구하기 위해서는 새로 추가된 3개의 게이트를 사용한다. 각 게이트는 삭제 게이트(delete gate), 입력 게이트(input gate), 출력 게이트(output gate)라고 부르며 이 3개의 게이트에는 공통적으로 $$sigmoid$$ 함수가 존재한다. (단연, 이전에 공부해서 알겠지만, $$sigmoid$$를 거치면 0과 1사이의 값이 나오게 되고 이 값들로 게이트를 조절한다.)

* 이하 식에서 $$\sigma$$는 시그모이드($$sigmoid$$)를 의미합니다.
* 이하 식에서 $$tanh$$는 하이퍼볼릭탄젠트 함수를 의미합니다.
* $$W_{xi},\;W_{xg},\;W_{xf},\;W_{xo}$$는 $$x_t$$와 함께 각 게이트에서 사용되는 4개의 가중치입니다.
* $$W_{hi},\;W_{hg},\;W_{hf},\;W_{ho}$$는 $$h_{t-1}$$와 함께 각 게이트에서 사용되는 4개의 가중치입니다.
* $$b_i,\;b_g,\;b_f,\;b_o$$는 각 게이트에서 사용되는 4개의 편향입니다.

### 1. 입력 게이트 (input gate)



<figure><img src="../.gitbook/assets/image (4).png" alt="" width="563"><figcaption></figcaption></figure>

$$
i_t = \sigma(W_{xi}x_t+W_{hi}h_{t-1}+b_i)\\g_t = tanh(W_{xg}x_t+W_{hg}h_{t-1}+b_g)
$$

**입력 게이트는 현재 정보를 기억하기 위한 게이트이다.** 현재 시점 $$t$$의 $$x$$값과 입력 게이트로 이어지는 가중치 $$W_{xi}$$를 곱한 값과 이전 시점 $$t-1$$의 은닉 상태가 입력 게이트로 이어지는 가중치 $$W_{hi}$$를 곱한 값을 더하여 $$sigmoid$$ 함수를 지나고 이를 $$i_t$$라고 합니다. 그리고 현재 시점 $$t$$의 $$x$$값과 입력 게이트로 이어지는 가중치 $$W_{xi}$$를 곱한 값과 이전 시점 $$t-1$$의 은닉 상태가 입력 게이트로 이어지는 가중치 $$W_{hg}$$를 곱한 값을 더하여 $$tanh$$를 지나며 이를 $$g_t$$라고 합니다. 결국 $$sigmoid$$ 함수를 지나 0과 1사이의 값과 $$tanh$$ 함수를 지나 -1과 1 사이의 값 2개가 나오게 되고 이 두개의 값들을 가지고 해당 시점에서 선택된 기억할 정보의 양을 정합니다.

### 2. 삭제 게이트 (delete gate)

<figure><img src="../.gitbook/assets/image (5).png" alt="" width="563"><figcaption></figcaption></figure>

$$
f_t = \sigma(W_{xf}x_t + W_{hf}h_{t-1}+b_f)
$$

삭제 게이트는 기억을 삭제하기 위한 게이트입니다. 현재 시점 $$t$$의 $$x$$값과 이전 시점 $$t-1$$의 은닉 상태가 $$sigmoid$$ 함수를 지나게 됩니다. 이 함수를 지나서 나오는 0과 1 사이의 값이 곧 삭제 과정을 거친 정보의 양을 의미한다. 0에 가까울수록 정보가 많이 삭제된 것이고, 1에 가까울수록 정보를 온전히 기억한 것이다.

### 3. 셀 상태 (장기 상태)

<figure><img src="../.gitbook/assets/image (6).png" alt="" width="563"><figcaption></figcaption></figure>

$$
C_t = f_t\otimes C_{t-1}+i_t\otimes g_t
$$

LSTM에서 셀 상태 $$C_t$$를 장기상태라고 부르기도 합니다. 삭제 게이트에서 일부 기억을 잃은 상태에서부터 시작을 하고 입력 게이트에서 구한 $$i_t,\;g_t$$이 2개의 값에 대해서 원소별 곱(entrywise product)을 진행합니다. 같은 크기의 두 행렬이 있을 때 같은 위치의 성분끼리 곱하는 것을 의미하고 위 수식에서 $$\otimes$$로 표현하고 현 시점에서 선택된 기억해야할 값을 의미하게 됩니다.

이후 입력게이트에서 선택된 기억을 삭제 게이트의 결과값과 더하며 이를 현재 시점 $$t$$의 셀 상태라고 하고 이 값은 다음 $$t-1$$시점의 LSTM 셀로 넘어가게 됩니다. 만약 삭제 게이트의 출력값인 $$f_t$$가 0이 된다면, 이전 시점의 셀 상태값인 $$C_{t-1}$$은 현재 시점의 셀 상태값을 결정하기 위한 영향력이 0이 되면서 오직 입력 게이트의 결과만이 현재 시점의 셀 상태값 $$C_t$$를 결정할 수 있습니다.

이것은 삭제 게이트가 완전히 닫히고 입력게이트를 연 상태를 의미합니다. 반대로 입력게이트의 $$i_t$$값을 0이라고 한다면 현재 시점의 셀 상태값 $$C_t$$은 오직 이전 시점의 셀 상태값 $$C_{t-1}$$의 값에만 의존합니다. 이는 입력게이트를 완전히 닫고 삭제 게이트만을 연 상태를 말하고 결과적으로 삭제 게이트는 이전 시점의 입력을 얼마나 반영할지를 의미하게 되며 입력게이트는 현재 시점의 입력을 얼마나 반영할 지 결정합니다.

### 4. 출력 게이트와 은닉 상태 (단기 상태)

<figure><img src="../.gitbook/assets/image (7).png" alt="" width="563"><figcaption></figcaption></figure>

$$
o_t = \sigma(W_{xo}x_t+W_{ho}h_{t-1}+b_o)\\h_t = o_t\otimes tanh(c_t)
$$

출력 게이트는 현재 시점 $$t$$의 $$x$$값과 이전 시점 $$t-1$$의 은닉 상태가 $$sigmoid$$ 함수를 지난 값입니다. 이 값은 현재 시점 $$t$$의 은닉 상태를 결정하는 일에 사용됩니다. 이 은닉 상태를 단기상태라고 하기도 하는데 이 값이 $$tanh$$ 함수를 지나 -1과 1사이의 값으로 변환이 됩니다. 해당 값은 출력 게이트의 값과 연산되면서 값이 걸러지는 효과가 발생합니다. 단기상태의 값은 또한 출력층으로도 향하게 됩니다.



## Ⅲ. Implementation

LSTM 역시 torch안에 내장되어 구현되어 있기 때문에 LSTM을 이용한 네이버 리뷰분류를 구현해보도록 하겠습니다.

{% code lineNumbers="true" %}
```python




```
{% endcode %}

