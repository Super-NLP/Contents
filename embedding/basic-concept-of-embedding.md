---
description: 임베딩에 대한 기본적인 개념을 알아봅시다.
---

# 🕷️ Basic Concept of Embedding

임베딩에 대한 이야기를 하기에 앞서 단어의 표현을 구분해보겠습니다.\
단어는 크게 국소표현 (Local Representation) 방법과 분산표현 (Distributed Representation) 방법으로 나뉩니다.&#x20;

* **국소표현방법 (Local Representation)** : 해당 단어 그 자체만 보고 특정값을 매핑하여 단어를 표현하는 방식으로 이산 표현 (Discrete Representation) 이라고 하기도 합니다.
* **분산표현방법 (Distributed Representation)** : 그 단어를 표현하고자 주변을 참고하여 표현하는 방식으로 연속 표현 (Continuous Representation) 이라고 하기도 합니다.

이것을 알기쉽게 카테고리화하여 보면 다음과 같이 분류할 수 있습니다.

<figure><img src="../.gitbook/assets/image.png" alt=""><figcaption><p>Category of Word Representation</p></figcaption></figure>

여기서는 크게 One-hot Encoding과 One-hot Vector, Bag-of-Words와 그의 확장인 DTM(또는 TDM), 여기서 가중치 개념이 들어간 TF-IDF에 대해 소개할 것이고 공통적으로 사용되는 벡터의 유사도에 대해서도 이야기하겠습니다.

## Ⅰ. Integer Labeled Encoding

### ⅰ. One Hot Encoding & One Hot Vector

단어를 컴퓨터가 인지할 수 있는 수치로 바꾸는 가장 간편한 방법은 벡터로 표현하는 것입니다. 그 중에서도 가장 기본적인 방법은 **OHE (One-Hot Encoding)** 방식입니다. 이 방식은 단 하나의 1과 나머지 수많은 0들로 표현된 인코딩 방식을 뜻하고 이렇게 만들어진 벡터를 **OHV (One-Hot Vector)**라고 합니다. OHV의 차원은 보통 전체 어휘집합(vocabulary)의 개수가 되며, 보통 그 숫자는 매우 큰 숫자를 가집니다. 전체 단어에 대한 OHV는 다음과 같이 표현할 수 있습니다.

$$
v \in \{0, 1\}^{|V|}, where\;v\;is\;one\;hot\;vector\;and\;|V|\;is\;a\;vocabulary\;size.
$$

OHV는 단어를 불연속적 심볼로 보고 이산 확률 변수로 나타냅니다. 이산 확률 변수는 그 값을 불연속적으로 (= 0, 1) 가질 수 밖에 없습니다. 즉, OHV는 이산 확률 분포로부터 뽑아낸 샘플이라고 할 수 있습니다. 이처럼 사전(dictionary)내의 각 단어를 OHE을 통해 vector로 나타낼 수 있습니다.

다만 이 표현방식은 여러가지 문제점을 가지게 됩니다. 앞전에 이야기했듯 **벡터의 차원이 너무 커지게 됩니다.** 각 벡터는 단 하나의 1을 가지고 나머지는 0으로 채워지는데 이것은 벡터의 너무 많은 부분이 희소벡터가 되버리는 것입니다. 희소벡터의 가장 큰 문제점은 벡터 간 연산시 결과값이 0이 된다는 것으로 서로 직교(orthogonal)하는 경우가 많아진다는 것을 의미합니다. 이것을 쉽게 말하면 ‘강아지’와 ‘개’라는 단어는 서로 유사한 단어임에도 벡터화 하여 계산할 시 이 둘의 유사도가 0에 가까워진다는 것을 의미합니다. **즉, 단어와 단어 사이의 유사성을 판단할 수 없습니다.**

OHE 방식은 이러한 한계점이 명확하기 때문에 효과적인 방법이 아닙니다.&#x20;

{% code lineNumbers="true" %}
```python
from konlpy.tag import Okt

okt = Okt()
tokens = okt.morphs("나는 자연어 처리를 배운다")
print(f"tokenizing results : {tokens}")

word2idx = {word:index for index, word in enumerate(set(tokens))}
print('vocab :',word2idx)

def one_hot_encoding(word, word_to_index):
  one_hot_vector = [0]*(len(word_to_index))
  index = word_to_index[word]
  one_hot_vector[index] = 1
  return one_hot_vector

one_hot_encoding("자연어", word2idx)
```
{% endcode %}

```
tokenizing results : ['나', '는', '자연어', '처리', '를', '배운다']
vocab : {'는': 0, '배운다': 1, '자연어': 2, '처리': 3, '를': 4, '나': 5}
[0, 0, 1, 0, 0, 0]
```

### ⅱ. 희소벡터와 밀집벡터









## Ⅱ. Count based Embedding

### ⅰ. BoW : Bag of Words





### ⅱ. DTM : Documents-Term Matrix





### ⅲ. TF-IDF : Term Frequency - Inverse Documents Frequency







## Ⅲ. Vector Similarity

### ⅰ. Cosine Similarity





### ⅱ. L1 Distance





### ⅲ. L2 Distance





### ⅳ. Infinity Norm





## Ⅳ. Word Embedding

단어를 밀집 벡터(dense vector)의 형태로 표현하는 방법을 **Word embedding(워드 임베딩)**이라고 합니다. 그리고 밀집 벡터를 **Word Embedding**하여 나온 결과를 **Embedding Vector(임베딩 벡터)**라고 합니다.

워드 임베딩 방법론으로는 LSA, Word2Vec, FastText, Glove 등이 있습니다. 유명한 딥러닝 프레임워크이자 커뮤니티인 파이토치에서 제공하는 도구인 `nn.embedding()`는 앞서 언급한 방법들을 사용하지는 않지만, 단어를 랜덤한 값을 가지는 밀집 벡터로 변환한 뒤에, 인공 신경망의 가중치를 학습하는 것과 같은 방식으로 단어 벡터를 학습하는 방법을 사용합니다. 아래의 표는 앞서 배운 원-핫 벡터와 지금 배우고 있는 임베딩 벡터의 차이를 보여줍니다.

<table><thead><tr><th width="132">-</th><th width="313">원-핫 벡터</th><th>임베딩 벡터</th></tr></thead><tbody><tr><td><strong>차원</strong></td><td>고차원 (단어 집합의 크기)</td><td>저차원</td></tr><tr><td><strong>다른 표현</strong></td><td>희소 벡터의 일종</td><td>밀집 벡터의 일종</td></tr><tr><td><strong>표현 방법</strong></td><td>수동</td><td>훈련 데이터로부터 학습함</td></tr><tr><td><strong>값의 타입</strong></td><td>1과 0</td><td>실수</td></tr></tbody></table>

이후부터는 본격적으로 Word Embedding에 대한 방법을 다뤄보겠습니다.
