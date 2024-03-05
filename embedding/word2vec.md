---
description: 본격적으로 단어를 Vector로 만드는 방법을 알아봅시다.
---

# 🕷️ Word2Vec

이제 우리는 신경망을 이용해서 단어의 벡터표현하는 방법을 알아보도록 하겠습니다. 가장 널리알려졌었던 것은 Word2Vec (2012) 이지만 사실 그 전에도 여러가지 시도는 이루어져 왔습니다. **잠재의미분석 (LSA : Latent Semantic Analysis)**과 **잠재 디리클레 할당 (LDA : Latent Dirichlet Allocation)**이라는 방법이 존재했었습니다. 그럼에도 <mark style="color:orange;background-color:yellow;">**이때 당시 Word2Vec이 주목을 받았던 이유는 신경망을 이용해서 낮은 계산성을 유지하는 효율성 때문**</mark>입니다.&#x20;

이러한 Word2Vec은 사실 **CBOW(Continous Bag of Words)**라는 방법과 **Skip-Gram**이라는 방법을 같이 부르는 말입니다. 각 방식에 대해 알아보고 이들의 한계점을 개선하기 위한 방법으로 제안된 Negative Sampling (2013) 을 이용한 **SGNS (Skip Gram with Negative Sampling)** 을 소개하겠습니다.

<figure><img src="../.gitbook/assets/image (5).png" alt="" width="563"><figcaption></figcaption></figure>

## Ⅰ. CBOW

앞서 말했듯 Word2Vec에는 CBOW(Continuous Bag of Words)와 Skip-Gram 두 가지 방식이 있습니다. 그중 **CBOW는 주변에 있는 단어들을 가지고, 중간에 있는 단어들을 예측하는 방법**입니다. 반대로, Skip-Gram은 중간에 있는 단어로 주변 단어들을 예측하는 방법입니다. 메커니즘 자체는 거의 동일하기 때문에 CBOW를 이해한다면 Skip-Gram도 손쉽게 이해 가능합니다. 예시를 들어 CBOW에 대해서 알아봅시다.

**예문 : "The fat cat sat on the mat"**

위와 같은 예시를 가지고 CBOW를 알아봅시다. 가운데 단어를 예측하는 것이 CBOW라고 했으니 {"The", "fat", "cat", "on", "the", "mat"}으로부터 **sat을 예측하는 것은 CBOW가 하는 일**입니다. 이 때 예측해야하는 단어 sat을 **중심 단어(center word)**라고 하고, 예측에 사용되는 단어들을 **주변 단어(context word)**라고 합니다.&#x20;

중심 단어를 예측하기 위해서 **앞, 뒤로 몇 개의 단어를 볼지**를 결정했다면 이 **범위를 윈도우(window)**라고 합니다. 예를 들어서 윈도우 크기가 2이고, 예측하고자 하는 중심 단어가 sat이라고 한다면 앞의 두 단어인 fat와 cat, 그리고 뒤의 두 단어인 on, the를 참고합니다. 윈도우 크기가 n이라고 한다면, 실제 중심 단어를 예측하기 위해 참고하려고 하는 주변 단어의 개수는 2n이 될 것입니다.

<figure><img src="../.gitbook/assets/image (6).png" alt=""><figcaption><p>window = 2</p></figcaption></figure>

윈도우를 계속 움직여서 주변 단어와 중심 단어 선택을 바꿔가며 학습을 위한 데이터 셋을 만들 수 있는데, 이 방법을 **슬라이딩 윈도우(sliding window)**라고 합니다.

<figure><img src="../.gitbook/assets/image (7).png" alt=""><figcaption><p>CBOW Mechanism</p></figcaption></figure>

위의 그림은 CBOW의 Mechanism을 도식화하여 나타낸 그림입니다. 입력층(Input layer)의 입력으로서 앞, 뒤로 사용자가 정한 윈도우 크기 범위 안에 있는 주변 단어들의 원-핫 벡터가 들어가게 되고, 출력층(Output layer)에서 예측하고자 하는 중간 단어의 원-핫 벡터가 필요합니다.

사실 Word2Vec은 딥 러닝 모델(Deep Learning Model)은 아닙니다. 보통 딥 러닝이라함은, 입력층과 출력층 사이의 은닉층의 개수가 충분히 쌓인 신경망을 학습할 때를 말하는데 Word2Vec의 경우 입력층과 출력층 사이에 하나의 은닉층만이 존재합니다. 이렇게 은닉층(hidden Layer)이 1개인 경우에는 일반적으로 심층신경망(Deep Neural Network)이 아니라 **얕은신경망(Shallow Neural Network)**이라고 부릅니다. 또한 Word2Vec의 은닉층은 일반적인 은닉층과는 달리 활성화 함수가 존재하지 않으며 **룩업 테이블**이라는 연산을 담당하는 층으로 일반적인 은닉층과 구분하기 위해 **투사층(projection layer)**이라고 부르기도 합니다.

<figure><img src="../.gitbook/assets/image (8).png" alt="" width="411"><figcaption></figcaption></figure>

위의 CBOW Mechanism에서 중요한 것은 2가지인데, 첫번째로 투사층(=Projection layer)의 크기가 M이라는 점입니다. 이러한 투사층의 크기 M은 임베딩하고 난 벡터의 차원이 됩니다. 즉, 위의 그림에서 투사층의 크기는 M=5이기 때문에 CBOW를 수행하고나서 얻는 각 단어의 임베딩 벡터의 차원은 5가 될 것입니다.

입력층과 투사층 사이의 가중치 W는 V × M 행렬이며, 투사층에서 출력층사이의 가중치 W'는 M × V 행렬이라는 점입니다. 여기서 **V는 단어 집합의 크기**를 의미합니다. 즉, 위의 그림처럼 원-핫 벡터의 차원이 7이고, M은 5라면 가중치 W는 7 × 5 행렬이고, W'는 5 × 7 행렬이 될 것입니다. 주의할 점은 이 두 행렬은 동일한 행렬을 전치(transpose)한 것이 아니라, 서로 다른 행렬이라는 점입니다. 덧붙여 인공 신경망의 훈련을 하기 전에는 이 가중치 행렬 W와 W'는 굉장히 작은 랜덤 값을 가지고 학습을 시작하게됩니다. **CBOW는 주변 단어로 중심 단어를 더 정확히 맞추기 위해 계속해서 이 W와 W'를 학습해가는 구조**입니다

<figure><img src="../.gitbook/assets/image (9).png" alt=""><figcaption></figcaption></figure>

입력으로 들어오는 주변 단어의 OHV와 가중치 W 행렬의 곱이 어떻게 이루어지는지 보겠습니다. 위의 그림에서는 각 주변 단어의 OHV를 $$x$$로 표기하였습니다. 입력 벡터는 OHV입니다. i번째 인덱스에 1이라는 값을 가지고 그 외의 0의 값을 가지는 입력 벡터와 가중치 W 행렬의 곱은 사실 W행렬의 i번째 행을 그대로 읽어오는 것과(lookup) 동일합니다. 그래서 이 작업을 **룩업 테이블(lookup table)**이라고 부릅니다. 앞서 CBOW의 목적은 W와 W'를 잘 훈련시키는 것이라고 언급한 적이 있는데, 사실 그 이유가 여기서 lookup해온 W의 각 행벡터가 사실 Word2Vec을 수행한 후의 각 단어의 M차원의 크기를 갖는 임베딩 벡터들이기 때문입니다.

<figure><img src="../.gitbook/assets/image (10).png" alt=""><figcaption></figcaption></figure>

이렇게 각 주변 단어의 OHV에 대해서 가중치 W가 곱해서 생겨진 결과 벡터들은 투사층에서 만나 이 벡터들의 평균인 벡터를 구하게 됩니다. 만약 윈도우 크기가 2라면, 입력 벡터의 총 개수는 2n이므로 중간 단어를 예측하기 위해서는 총 4개가 입력 벡터로 사용됩니다. 그렇기 때문에 평균을 구할 때는 4개의 결과 벡터에 대해서 평균을 구하게 됩니다. **투사층에서 벡터의 평균을 구하는 부분은 CBOW가 Skip-Gram과 다른 차이점**이기도 합니다. 나중에 다루겠지만 Skip-Gram은 입력이 중심 단어 하나이기때문에 투사층에서 벡터의 평균을 구하지 않습니다.

<figure><img src="../.gitbook/assets/image (11).png" alt=""><figcaption></figcaption></figure>

이렇게 구해진 평균 벡터는 두번째 가중치 행렬 W'와 곱해집니다. 곱셈의 결과로는 OHV들과 차원이 V로 동일한 벡터가 나옵니다. 만약 입력 벡터의 차원이 7이었다면 여기서 나오는 벡터도 마찬가지입니다.

이 벡터에 CBOW는 **소프트맥스(softmax) 함수**를 취하는데, 소프트맥스 함수로 인한 출력값은 0과 1사이의 실수로, 각 원소의 총 합은 1이 되는 상태로 바뀝니다. 이렇게 나온 벡터를 **스코어 벡터(score vector)**라고 합니다. 스코어 벡터의 각 차원 안에서의 값이 의미하는 것은 아래와 같습니다.

스코어 벡터의 j번째 인덱스가 가진 0과 1사이의 값은 j번째 단어가 중심 단어일 확률을 나타냅니다. 그리고 이 스코어 벡터는 우리가 실제로 값을 알고있는 벡터인 중심 단어 원-핫 벡터의 값에 가까워져야 합니다. 스코어 벡터를 $$\hat{y}$$라고 하겠습니다. 중심 단어를 $$y$$로 했을 때, 이 두 벡터값의 오차를 줄이기위해 **CBOW는 손실 함수(loss function)로 cross-entropy 함수를 사용**하여 역전파(Back-Propagatio)를 수행하게 됩니다. 역전파가 진행됨에 따라 W와 W'가 학습이 되고, 학습이 다 되었다면 M차원의 크기를 갖는 W의 행이나 W'의 열로부터 어떤 것을 임베딩 벡터로 사용할지를 결정하면 됩니다. 때로는 W와 W'의 평균치를 가지고 임베딩 벡터를 선택하기도 합니다.

## Ⅱ. Skip-Gram

Skip-gram은 CBOW를 이해했다면, 메커니즘 자체는 동일하기 때문에 쉽게 이해할 수 있습니다. 앞서 CBOW에서는 주변 단어를 통해 중심 단어를 예측했다면, Skip-gram은 중심 단어에서 주변 단어를 예측하려고 합니다.

<figure><img src="../.gitbook/assets/image (12).png" alt="" width="351"><figcaption><p>Skip-Gram</p></figcaption></figure>

앞서 언급한 동일한 예문에 대해서 인공 신경망을 도식화해보면 위와 같습니다. 이제 중심 단어에 대해서 주변 단어를 예측하기 때문에, 투사층에서 벡터들의 평균을 구하는 과정은 없습니다. 여러 논문에서 성능 비교를 진행했을 때, 전반적으로 Skip-gram이 CBOW보다 성능이 좋다고 알려져 있습니다.

즉, CBOW과 Skip-gram은 매우 유사한 구조를 가지지만 입력(input)과  출력(output)이 반대인 모델이라 할 수 있습니다. 이 모델들은 주어진 단어를 N차원의 벡터로 projection한 뒤 softmax 함수를 이용해 출력단어를 맞추도록 학습하는 것입니다.

## Ⅲ. SGNS

이제까지 공부한 Word2Vec 모델에는 한 가지 문제점이 있습니다. 바로 속도입니다. Word2Vec의 마지막 단계를 보면 출력층에 있는 소프트맥스 함수는 단어 집합 크기의 벡터 내의 모든 값을 0과 1사이의 값이면서 모두 더하면 1이 되도록 바꾸는 작업을 수행합니다. 그리고 이에 대한 오차를 구하고 모든 단어에 대한 임베딩을 조정합니다. 그 단어가 중심 단어나 주변 단어와 전혀 상관없는 단어라도 마찬가지 입니다. 그런데 만약 단어 집합의 크기가 수백만에 달한다면 이 작업은 굉장히 무거운 작업입니다. 여기서 중요한 건 Word2Vec이 모든 단어 집합에 대해서 소프트맥스 함수를 수행하고, 역전파를 수행하므로 주변 단어와 상관 없는 모든 단어까지의 워드 임베딩 조정 작업을 수행한다는 겁니다. 만약 마지막 단계에서 '강아지'와 '고양이'와 같은 단어에 집중하고 있다면, Word2Vec은 사실 '돈가스'나 '컴퓨터'와 같은 연관 관계가 없는 수많은 단어의 임베딩을 조정할 필요가 없습니다.

이러한 과도한 계산량의 문제를 해결하기 위해 Hierarchical Softmax와 Negative Sampling이라는 2가지 방식이 제안되었습니다. 여기서는 Negative Sampling만 다루겠습니다.

기존의 Softmax함수를 계산하지않고 다른 방식을 통해서 Word Embedding을 시켜보는 것이 Negative Sampling의 접근법입니다. 어떠한 corpus상에서 등장한 단어의 개수가 N개라면, Softmax 함수는 이 N개를 모두 고려하여 확률값을 계산하지만 '특정 맥락에 나와야 하는 단어 몇개, 나오면 안되는 단어 몇개'를 출려서 계산하는 접근법인것 입니다.

학습속도 향상을 위해 제시된 2가지 방식을 놓고 평가한 결과 Hierarchical Softmax보다는 Negative Sampling이 더 좋은 성능을 보였으며 2014년 이후부터는 이것과 Skip-gram을 합친 "SGNS : Skip Gram with Negative Sampling"으로 불리고 Word2Vec을 대신하기도 합니다. 이때부터 GloVe(2014)와 함께 Word Embedding의 양대 산맥으로 자리매김하게됩니다.

***

`gensim` 라이브러리를 이용해 네이버 영화 리뷰 데이터로 한국어 Word2Vec을 만들어봅시다.

{% code lineNumbers="true" %}
```python
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Mecab
from tqdm import tqdm

urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt")
```
{% endcode %}

```
('ratings.txt', <http.client.HTTPMessage at 0x7aef1ce46890>)
```

{% code lineNumbers="true" %}
```python
# 데이터 로드 확인
train_data = pd.read_table('ratings.txt')
train_data
```
{% endcode %}

<figure><img src="../.gitbook/assets/image (13).png" alt="" width="563"><figcaption></figcaption></figure>

{% code lineNumbers="true" %}
```python
print(len(train_data)) # 리뷰 개수 출력
```
{% endcode %}

```
200000
```

{% code lineNumbers="true" %}
```python
# NULL 값 존재 유무에 따라 결측치 제거
if train_data.isnull().values.any():
    train_data = train_data.dropna(how = 'any') # Null 값이 존재하는 행 제거
    print(train_data.isnull().values.any()) # Null 값이 존재하는지 확인

print(len(train_data)) # 리뷰 개수 출력 (결측치 제거 후)
```
{% endcode %}

```
False
199992
```

{% code lineNumbers="true" %}
```python
# 정규 표현식을 통한 한글 외 문자 제거
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
train_data[:5] # 상위 5개 출력
```
{% endcode %}

<figure><img src="../.gitbook/assets/image (14).png" alt="" width="563"><figcaption></figcaption></figure>

{% code lineNumbers="true" %}
```python
# 불용어 정의
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

# 형태소 분석기를 사용한 토큰화 작업 (다소 시간 소요)
m = Mecab()

tokenized_data = []
for sentence in tqdm(train_data['document']):
    tokenized_sentence = m.morphs(sentence) # 토큰화
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
    tokenized_data.append(stopwords_removed_sentence)
```
{% endcode %}

```
100%|██████████| 199992/199992 [00:20<00:00, 9566.99it/s]
```

{% code lineNumbers="true" %}
```python
# 리뷰 길이 분포 확인
print('리뷰의 최대 길이 :',max(len(review) for review in tokenized_data))
print('리뷰의 평균 길이 :',sum(map(len, tokenized_data))/len(tokenized_data))
plt.hist([len(review) for review in tokenized_data], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()
```
{% endcode %}

<figure><img src="../.gitbook/assets/image (15).png" alt="" width="563"><figcaption></figcaption></figure>

{% code lineNumbers="true" %}
```python
from gensim.models import Word2Vec

model = Word2Vec(sentences = tokenized_data, vector_size = 100, window = 5, min_count = 5, workers = 4, sg = 0)
# 완성된 임베딩 매트릭스의 크기 확인
print(model.wv.vectors.shape)
# 유사단어 확인
print(model.wv.most_similar("최민식"))
print(model.wv.most_similar("히어로"))
```
{% endcode %}

```
(18139, 100)
[('드니로', 0.8695348501205444), ('안성기', 0.8506764769554138), ('한석규', 0.8406697511672974), ('박중훈', 0.8380630016326904), ('전도연', 0.8284204006195068), ('박신양', 0.8283261656761169), ('이병헌', 0.8181988596916199), ('단역', 0.8171784281730652), ('윤제문', 0.815010130405426), ('클리프', 0.813610315322876)]
[('슬래셔', 0.879467785358429), ('호러', 0.8555783033370972), ('정통', 0.8374928832054138), ('느와르', 0.8341838121414185), ('패러디', 0.8325538039207458), ('블록버스터', 0.8305109739303589), ('원조', 0.817631185054779), ('무협', 0.8109155893325806), ('무비', 0.8089743852615356), ('첩보', 0.8024275898933411)]
```

{% code lineNumbers="true" %}
```python
from gensim.models import KeyedVectors

model.wv.save_word2vec_format('kor_review_w2v') # 모델 저장
loaded_model = KeyedVectors.load_word2vec_format("kor_review_w2v") # 모델 로드

print(loaded_model.most_similar("히어로"))
```
{% endcode %}

```
[('슬래셔', 0.879467785358429), ('호러', 0.8555783033370972), ('정통', 0.8374928832054138), ('느와르', 0.8341838121414185), ('패러디', 0.8325538039207458), ('블록버스터', 0.8305109739303589), ('원조', 0.817631185054779), ('무협', 0.8109155893325806), ('무비', 0.8089743852615356), ('첩보', 0.8024275898933411)]
```

***

자연어 처리 작업을 할때, 훈련 데이터로부터 처음부터 임베딩 벡터를 훈련시키기도 하지만, 위키피디아 등의 방대한 데이터로 사전에 훈련된 워드 임베딩(pre-trained word embedding vector)를 가지고 와서 해당 벡터들의 값을 원하는 작업에 사용 할 수도 있습니다. 예를 들어서 감성 분류 작업을 하는데 훈련 데이터의 양이 부족한 상황이라면, 다른 방대한 데이터를 Word2Vec이나 GloVe 등으로 사전에 학습시켜놓은 임베딩 벡터들을 가지고 와서 모델의 입력으로 사용하는 것이 때로는 더 좋은 성능을 얻을 수 있습니다. 사전 훈련된 워드 임베딩을 가져와서 간단히 단어들의 유사도를 구해보는 실습을 해보겠습니다. 이번에는 사전학습된 Word2Vec 모델을 가져와 다뤄보도록 하겠습니다.

{% code lineNumbers="true" %}
```python
import gensim
import urllib.request

# 구글의 사전 훈련된 Word2Vec 모델을 로드.
# Download Link : https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

print(word2vec_model.vectors.shape)
print(word2vec_model.similarity('this', 'is'))
print(word2vec_model.similarity('post', 'book'))
print(word2vec_model['book'])
```
{% endcode %}

```
// Some code
```

Word2vec 모델은 자연어 처리에서 단어를 밀집 벡터로 만들어주는 단어 임베딩 방법론이지만 최근에 들어서는 자연어 처리를 넘어서 추천 시스템에도 사용되고 있는 모델입니다. 적당하게 데이터를 나열해주면 Word2vec은 위치가 근접한 데이터를 유사도가 높은 벡터를 만들어준다는 점에서 착안된 아이디어입니다. 관심있는 분들은 구글에 'item2vec'을 찾아보시길 바랍니다.
