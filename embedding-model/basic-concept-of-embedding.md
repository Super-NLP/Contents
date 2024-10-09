---
description: 임베딩에 대한 기본적인 개념을 알아봅시다.
---

# 🕷️ Basic Concept of Embedding

임베딩에 대한 이야기를 하기에 앞서 단어의 표현을 구분해보겠습니다.\
단어는 크게 국소표현 (Local Representation) 방법과 분산표현 (Distributed Representation) 방법으로 나뉩니다.&#x20;

* **국소표현방법 (Local Representation)** : 해당 단어 그 자체만 보고 특정값을 매핑하여 단어를 표현하는 방식으로 이산 표현 (Discrete Representation) 이라고 하기도 합니다.
* **분산표현방법 (Distributed Representation)** : 그 단어를 표현하고자 주변을 참고하여 표현하는 방식으로 연속 표현 (Continuous Representation) 이라고 하기도 합니다.

이것을 알기쉽게 카테고리화하여 보면 다음과 같이 분류할 수 있습니다.

<figure><img src="../.gitbook/assets/image (4) (1).png" alt=""><figcaption><p>Category of Word Representation</p></figcaption></figure>

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

#### 1. 희소 표현 (Sparse Representation)

앞서말한 OHE을 통해서 나온 OHV들은 표현하고자 하는 단어의 인덱스의 값만 1이고, 나머지 인덱스에는 전부 0으로 표현되는 벡터의 표현 방법이었습니다. 이렇게 벡터(vector) 또는 행렬(matrix)의 값이 대부분이 0으로 표현되는 방법을 희소 표현(sparse representation)이라고 합니다. 이렇게 만들어진 벡터를 **희소 벡터(Sparse Vector)**라고 합니다. 다시말해 **OHV는 희소 벡터(sparse vector)**입니다.

이러한 희소 벡터의 문제점은 <mark style="color:red;">**단어의 개수가 늘어나면 벡터의 차원이 한없이 커진다는 점**</mark>입니다. OHV로 표현할 때는 갖고 있는 corpus에 단어가 10,000개였다면 이 벡터의 차원은 10,000이어야만 했습니다. 심지어 그 중에서 단어의 인덱스에 해당되는 부분만 1이고 나머지는 0의 값을 가져야만 했습니다. 단어 집합이 클수록 고차원의 벡터가 됩니다. 예를 들어 단어가 10,000개 있고 인덱스가 0부터 시작하면서 강아지란 단어의 인덱스는 4였다면 OHV는 이렇게 표현되어야 했습니다.

```
Ex) 강아지 = [ 0 0 0 0 1 0 0 0 0 0 0 0 ... 중략 ... 0]

# 이때 1 뒤의 0의 수는 9995개.
```

즉, 공간적 낭비를 불러일으킵니다. OHV뿐만 아니라 희소 표현의 일종인 DTM과 같은 경우에도 특정 문서에 여러 단어가 다수 등장하였으나, 다른 많은 문서에서는 해당 특정 문서에 등장했던 단어들이 전부 등장하지 않는다면 역시나 행렬의 많은 값이 0이 되면서 공간적 낭비를 일으킵니다. 이러한 관점에서 **DTM도 희소 벡터**입니다. OHV와 같은 <mark style="color:red;background-color:red;">**희소 벡터의 문제점은 단어의 의미를 표현하지 못한다는 점입니다.**</mark>

#### 2. 밀집 표현(Dense Representation)

이러한 희소 표현과 반대되는 개념으로 **밀집 표현(Dense Representation)이** 있습니다. 밀집 표현은 벡터의 차원을 단어 집합의 크기로 상정하지 않습니다. 사용자가 설정한 값으로 모든 단어의 벡터 표현의 차원을 맞춥니다. 또한, 이 과정에서 더 이상 **0과 1만 가진 값이 아니라 실수값**을 가지게 됩니다. 희소표현때 들었던 예시를 임의로 128차원으로 설정해서 표현해보면 아래와 같습니다ㅏ.

```
Ex) 강아지 = [0.2 1.8 1.1 -2.1 1.1 2.8 ... 중략 ...] 

# 이 벡터의 차원은 128
```

즉, 모든 단어의 벡터 표현의 차원은 128로 바뀌면서 모든 값이 실수가 됩니다. 이러한 결과로 벡터의 차원이 밀집되었다고 하여 **밀집 벡터(dense vector)**라고 합니다.

## Ⅱ. Count based Embedding

### ⅰ. BoW : Bag of Words

Bag of Words는 단어들의 순서는 전혀 고려하지 않고, 단어들의 출현 빈도(frequency)에만 집중하는 임베딩 방법입니다. 간단한 방식인만큼 바로 구현예제를 보면 바로 알 수 있습니다.

{% code lineNumbers="true" %}
```python
from konlpy.tag import Okt

okt = Okt()

def build_bag_of_words(document):
  # 형태소 분석 및 데이터 정제
  document = document.replace('.', '')
  toked_document = okt.morphs(document)

  word_to_index = {}
  bow = []

  for word in toked_document:  
    if word not in word_to_index.keys():
      word_to_index[word] = len(word_to_index)  
      # BoW에 전부 기본값 1을 넣는다.
      bow.insert(len(word_to_index) - 1, 1)
    else:
      # 재등장하는 단어의 인덱스
      index = word_to_index.get(word)
      # 재등장한 단어는 해당하는 인덱스의 위치에 1을 더한다.
      bow[index] = bow[index] + 1

  return word_to_index, bow

doc1 = "정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다."
vocab, bow = build_bag_of_words(doc1)
print('Vocabulary :', vocab)
print('BoW vector :', bow)
```
{% endcode %}

```
Vocabulary : {'정부': 0, '가': 1, '발표': 2, '하는': 3, '물가상승률': 4, '과': 5, '소비자': 6, '느끼는': 7, '은': 8, '다르다': 9}
BoW vector : [1, 2, 1, 1, 2, 1, 1, 1, 1, 1]
```

### ⅱ. DTM : Documents-Term Matrix

서로 다른 문서들의 BoW들을 결합한 표현 방법인 문서 단어 행렬(Document-Term Matrix, DTM) 표현 방법을 다뤄봅시다. 행과 열을 반대로 선택하면 TDM이라고 부르기도 하는데 같은 의미로써 사용됩니다. DTM을 가지고 서로 다른 문서들을 비교할 수 있습니다.

<mark style="color:green;">**DTM**</mark>**은 다수의 문서에서 등장하는 각 단어들의 빈도수를 행렬로 표현한 것**을 말합니다. 즉, <mark style="background-color:green;">**각 문서에 대한 BoW를 하나의 행렬로 만든 것**</mark>으로 생각할 수 있으며, BoW와 다른 표현 방법이 아닌 확장의 개념으로 이해하는게 좋을 것 같습니다. 예를 들어 아래와 같은 4개의 문서가 있다고 합시다.

* **문서1 : 먹고 싶은 사과**
* **문서2 : 먹고 싶은 바나나**
* **문서3 : 길고 노란 바나나 바나나**
* **문서4 : 저는 과일이 좋아요**

각각에 대해 띄어쓰기 단위 토큰화를 수행한다고 가정하고, 문서 단어 행렬로 표현하면 다음과 같습니다.

<figure><img src="../.gitbook/assets/image (1) (1).png" alt="" width="375"><figcaption><p>DTM 예시</p></figcaption></figure>

### ⅲ. TF-IDF : Term Frequency - Inverse Documents Frequency

DTM 역시 BoW의 확장개념인만큼 본질적으로는 희소 행렬의 문제점을 고스란히 안고 있습니다.\
즉, 사용한 corpus의 크기가 크면 클수록 차원이 매우 크게 증가한다는 점과 높은 계산복잡도를 가진다는 점입니다. 또한 빈도수를 기반으로 하였기에 불용어와 같이 높은 빈도수를 가지나 자연어처리에서 큰 의미를 가지지 못할 경우가 포함될 수 있다는 말입니다. 이를 해결하기 위해 <mark style="background-color:purple;">**DTM에 불용어와 중요한 단어에 대해서 가중치를 줄 수 있는 방법**</mark>을 고안하게 되었습니다. <mark style="color:purple;">**TF-IDF**</mark>를 살펴봅시다.

TF-IDF (Term Frequency-Inverse Document Frequency)는 "단어의 빈도"와 "역 문서 빈도"(문서의 빈도에 특정 식을 취함)를 사용하여 DTM 내의 **각 단어들마다 중요한 정도를 가중치로 주는 방법**입니다. 우선 DTM을 만든 후, TF-IDF 가중치를 부여하여 구할 수 있습니다. 이 방법은 주로 문서의 유사도를 구하는 작업, 검색 시스템에서 검색 결과의 중요도를 정하는 작업, 문서 내에서 특정 단어의 중요도를 구하는 작업 등에 쓰일 수 있습니다. TF-IDF는 TF와 IDF를 곱한 값을 의미하는데 이를 식으로 표현해보겠습니다. 문서를 $$d$$, 단어를 $$t$$, 문서의 총 개수를 $$n$$이라고 표현할 때 $$TF, DF, IDF$$는 각각 다음과 같이 정의할 수 있습니다.

* $$tf(d,t)$$ : 특정 문서 $$d$$에서의 특정 단어 $$t$$의 등장 횟수
  * tf값은 앞선 DTM을 다룰때 이미 구하였습니다. DTM 자체가 문서에서 각 단어의 등장하는 빈도수 값을 의미하기 때문입니다.
* $$df(t)$$ : 특정 단어 t가 등장한 문서의 수
  * 여기서 특정 단어가 각 문서, 또는 문서들에서 몇 번 등장했는지는 관심가지지 않고 오직 **특정 단어 t가 등장한 문서의 수**에만 관심을 가집니다. 즉, 어떤 단어가 몇개의 문서에 등장했는지에만 주목하는 것입니다. 하나의 문서에 특정단어가 여러번 등장했어도 df값은 1이 됩니다.
* $$idf(t)$$ : $$df(t)$$에 반비례하는 수

$$
idf(t) = log({n \over {1+df(t)}})
$$

idf는 위와 같은 수식을 가지게 됩니다. 사실 idf는 df에 역수를 취해주고 싶은게 중요합니다. $$log()$$를 취해준 식의 분모에 1을 더해준 것은 $$log$$를 사용하지 않았을 때 총 문서 수인 $$n$$이 커지면 커질수록 idf값이 너무나도 커져버리기 때문에 사용합니다. 이는 불용어와 같이 자주 사용되는 단어들이 그렇지 않은 단어들에 비해 등장횟수가 생각보다 크게 차이나기 때문에 잘 등장하지 않는 희귀단어에 엄청난 가중치가 부여될 수 있는 현상을 막기 위함이기도 합니다. 분모에 1을 더해준 이유는 특정한 단어가 문서에 등장하지 않을경우 분모가 0이 되는것을 방지하기 위함입니다.

TF-IDF는 모든 문서에서 자주 등장하는 단어는 중요도가 낮다고 판단하며, 특정 문서에서만 자주 등장하는 단어는 중요도가 높다고 판단합니다. 이 값이 낮으면 중요도가 낮은 것이며, 값이 크면 중요도가 큰 것입니다. 즉, 우리가 자주 사용하게 되는 the나 a와 같이 불용어의 경우에는 보통 문서에 자주 등장하기때문에 자연스럽게 불용어의 TF-IDF의 값은 다른 단어의 TF-IDF에 비해서 낮아지게 됩니다. 이것을 직접 구현해봅시다.

{% code lineNumbers="true" %}
```python
import pandas as pd # 데이터프레임 사용을 위해
from math import log # IDF 계산을 위해

docs = [
  '먹고 싶은 사과',
  '먹고 싶은 바나나',
  '길고 노란 바나나 바나나',
  '저는 과일이 좋아요'
] 
vocab = list(set(w for doc in docs for w in doc.split()))
vocab.sort()

# 총 문서의 수
N = len(docs) 

def tf(t, d):
  return d.count(t)

def idf(t):
  df = 0
  for doc in docs:
    df += t in doc
  return log(N/(df+1))

def tfidf(t, d):
  return tf(t,d)* idf(t)

result = []

# 각 문서에 대해서 아래 연산을 반복
for i in range(N):
  result.append([])
  d = docs[i]
  for j in range(len(vocab)):
    t = vocab[j]
    result[-1].append(tf(t, d))

tf_ = pd.DataFrame(result, columns = vocab)
tf_
```
{% endcode %}

<figure><img src="../.gitbook/assets/image (1) (1) (1).png" alt="" width="375"><figcaption><p>tf_</p></figcaption></figure>

{% code lineNumbers="true" %}
```python
result = []
for j in range(len(vocab)):
    t = vocab[j]
    result.append(idf(t))

idf_ = pd.DataFrame(result, index=vocab, columns=["IDF"])
idf_
```
{% endcode %}

<figure><img src="../.gitbook/assets/image (2) (1).png" alt="" width="139"><figcaption><p>idf</p></figcaption></figure>

{% code lineNumbers="true" %}
```python
result = []
for i in range(N):
  result.append([])
  d = docs[i]
  for j in range(len(vocab)):
    t = vocab[j]
    result[-1].append(tfidf(t,d))

tfidf_ = pd.DataFrame(result, columns = vocab)
tfidf_
```
{% endcode %}

<figure><img src="../.gitbook/assets/image (3) (1).png" alt="" width="563"><figcaption><p>tf-idf</p></figcaption></figure>

## Ⅲ. Vector Similarity

우리는 이제까지 자연어를 벡터형태로 계속해서 바꿔왔습니다. 이렇게 형태를 바꾼데에는 의미적인 해석을 컴퓨터가 더 잘하게끔 하기 위함인데요, 가장 기본적인 벡터간 유사도를 측정하는 방법을 소개하겠습니다.

### ⅰ. Cosine Similarity

코사인 유사도 함수는 두 벡터 사이의 방향과 크기를 모두 고려하는 방법입니다.

{% code lineNumbers="true" %}
```python
def get_cosine_similarity(x1, x2):
    return (x1 * x2).sum() / ((x1**2).sum()**.5 * (x2**2).sum()**.5)
```
{% endcode %}

### ⅱ. L1 Distance

L1 Diastance는 L1 Norm을 사용한 것으로 **Manhatten distance**라고도 합니다. 이 방법은 두 벡터의 각 차원별 값의 차이의 절대값을 모두 합한 값을 사용합니다.

$$
d_{L1}(w,v)=\sum_{i=1}^{d}|w_i - v_i|, \;\;\;where\;\;w,v \in \mathbb{R}^d.
$$

{% code lineNumbers="true" %}
```python
import torch

def get_li_distance(x1, x2):
    return ((x1-x2).abs()).sum()
```
{% endcode %}

### ⅲ. L2 Distance

L2 Distance는 **Euclidean distance**라고 하며 우리가 잘 알고있는 좌표계에서 거리를 측정하는 방법을 말하고 차원별 값 차이의 제곱의 합에 루트를 취한 형태를 가집니다.

$$
d_{L2}(w,v)=\sqrt{\sum_{i=1}^{d}|w_i - v_i|^2}, \;\;\;where\;\;w,v \in \mathbb{R}^d.
$$

{% code lineNumbers="true" %}
```python
import torch

def get_l2_distance(x1, x2):
    return ((x1-x2)**2).sum()**.5
```
{% endcode %}

### ⅳ. Infinity Norm

L1, L2 Distance가 있다면 $$L_\infin$$ Distance도 있습니다. 이 거리는 차원별 값의 차이 중 가장 큰 값을 나타냅니다.

$$
d_\infin(w,v)=max(|w_1-v_1|, |w_2-v_2|,\cdots,|w_d-v_d|), \;\;\;where\;\;w,v \in \mathbb{R}^d.
$$

{% code lineNumbers="true" %}
```python
def get_infinity_distance(x1, x2):
    return ((x1, x2).abs()).max()
```
{% endcode %}

## Ⅳ. Word Embedding

단어를 밀집 벡터(dense vector)의 형태로 표현하는 방법을 **Word embedding(워드 임베딩)**이라고 합니다. 그리고 밀집 벡터를 **Word Embedding**하여 나온 결과를 **Embedding Vector(임베딩 벡터)**라고 합니다.

워드 임베딩 방법론으로는 LSA, Word2Vec, FastText, Glove 등이 있습니다. 유명한 딥러닝 프레임워크이자 커뮤니티인 파이토치에서 제공하는 도구인 `nn.embedding()`는 앞서 언급한 방법들을 사용하지는 않지만, 단어를 랜덤한 값을 가지는 밀집 벡터로 변환한 뒤에, 인공 신경망의 가중치를 학습하는 것과 같은 방식으로 단어 벡터를 학습하는 방법을 사용합니다. 아래의 표는 앞서 배운 원-핫 벡터와 지금 배우고 있는 임베딩 벡터의 차이를 보여줍니다.

<table><thead><tr><th width="132">-</th><th width="313">원-핫 벡터</th><th>임베딩 벡터</th></tr></thead><tbody><tr><td><strong>차원</strong></td><td>고차원 (단어 집합의 크기)</td><td>저차원</td></tr><tr><td><strong>다른 표현</strong></td><td>희소 벡터의 일종</td><td>밀집 벡터의 일종</td></tr><tr><td><strong>표현 방법</strong></td><td>수동</td><td>훈련 데이터로부터 학습함</td></tr><tr><td><strong>값의 타입</strong></td><td>1과 0</td><td>실수</td></tr></tbody></table>

이후부터는 본격적으로 Word Embedding에 대한 방법을 다뤄보겠습니다.
