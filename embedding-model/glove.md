---
description: GloVe(2014)와 FastText(2016)에 대해 알아봅시다.
---

# 🕷️ GloVe

## Ⅰ. GloVe

글로브(Global Vectors for Word Representation, GloVe)는 카운트 기반과 예측 기반을 모두 사용하는 방법론으로 2014년에 미국 스탠포드대학에서 개발한 단어 임베딩 방법론입니다. 앞서 학습하였던 기존의 카운트 기반의 LSA(Latent Semantic Analysis)와 예측 기반의 Word2Vec의 단점을 지적하며 이를 보완한다는 목적으로 나왔고, 실제로도 Word2Vec만큼 뛰어난 성능을 보여줍니다. 현재까지의 연구에 따르면 단정적으로 Word2Vec와 GloVe 중에서 어떤 것이 더 뛰어나다고 말할 수는 없고, 이 두 가지 전부를 사용해보고 성능이 더 좋은 것을 사용하는 것이 바람직합니다.

{% code lineNumbers="true" %}
```bash
!pip install glove-python3
```
{% endcode %}

{% code lineNumbers="true" %}
```python
from glove import Corpus, Glove
import re
import urllib.request
import zipfile
from lxml import etree
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize

# 데이터 다운로드
urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/09.%20Word%20Embedding/dataset/ted_en-20160408.xml", filename="ted_en-20160408.xml")

targetXML = open('ted_en-20160408.xml', 'r', encoding='UTF8')
target_text = etree.parse(targetXML)

# xml 파일로부터 <content>와 </content> 사이의 내용만 가져온다.
parse_text = '\n'.join(target_text.xpath('//content/text()'))

# 정규 표현식의 sub 모듈을 통해 content 중간에 등장하는 (Audio), (Laughter) 등의 배경음 부분을 제거.
# 해당 코드는 괄호로 구성된 내용을 제거.
content_text = re.sub(r'\([^)]*\)', '', parse_text)

# 입력 코퍼스에 대해서 NLTK를 이용하여 문장 토큰화를 수행.
sent_text = sent_tokenize(content_text)

# 각 문장에 대해서 구두점을 제거하고, 대문자를 소문자로 변환.
normalized_text = []
for string in sent_text:
     tokens = re.sub(r"[^a-z0-9]+", " ", string.lower())
     normalized_text.append(tokens)

# 각 문장에 대해서 NLTK를 이용하여 단어 토큰화를 수행.
result = [word_tokenize(sentence) for sentence in normalized_text]
print('총 샘플의 개수 : {}'.format(len(result)))
```
{% endcode %}

```
[nltk_data] Downloading package punkt to /root/nltk_data...
[nltk_data]   Unzipping tokenizers/punkt.zip.
총 샘플의 개수 : 273424
```

{% code lineNumbers="true" %}
```python
corpus = Corpus() 
corpus.fit(result, window=5)
# 훈련 데이터로부터 GloVe에서 사용할 동시 등장 행렬 생성

glove = Glove(no_components=100, learning_rate=0.05)
glove.fit(corpus.matrix, epochs=20, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)
# 학습에 이용할 쓰레드의 개수는 4로 설정, 에포크는 20.
```
{% endcode %}

```
Performing 20 training epochs with 4 threads
Epoch 0
Epoch 1
Epoch 2
Epoch 3
Epoch 4
Epoch 5
Epoch 6
Epoch 7
Epoch 8
Epoch 9
Epoch 10
Epoch 11
Epoch 12
Epoch 13
Epoch 14
Epoch 15
Epoch 16
Epoch 17
Epoch 18
Epoch 19
```

{% code lineNumbers="true" %}
```python
model_result1=glove.most_similar("man")
print(model_result1)
```
{% endcode %}

```
[('woman', 0.9604732446102091), ('guy', 0.8963648521480801), ('girl', 0.8755874049593755), ('boy', 0.8466197641507356)]
```

## Ⅱ. FastText

단어를 벡터로 만드는 또 다른 방법으로는 페이스북에서 개발한 FastText가 있습니다. Word2Vec 이후에 나온 것이기 때문에, 메커니즘 자체는 Word2Vec의 확장이라고 볼 수 있습니다. Word2Vec와 FastText와의 가장 큰 차이점이라면 Word2Vec는 단어를 쪼개질 수 없는 단위로 생각한다면, FastText는 하나의 단어 안에도 여러 단어들이 존재하는 것으로 간주합니다. 내부 단어. 즉, 서브워드(subword)를 고려하여 학습합니다.

FastText에서는 각 단어는 글자 단위 n-gram의 구성으로 취급합니다. n을 몇으로 결정하는지에 따라서 단어들이 얼마나 분리되는지 결정됩니다. 예를 들어서 n을 3으로 잡은 트라이그램(tri-gram)의 경우, apple은 app, ppl, ple로 분리하고 이들을 벡터로 만듭니다. 더 정확히는 시작과 끝을 의미하는 <, >를 도입하여 아래의 5개 내부 단어(subword) 토큰을 벡터로 만듭니다.&#x20;

만약n = 3인 경우 \<ap, app, ppl, ple, le> 과 같이 쪼개집니다. 그리고 여기에 추가적으로 하나를 더 벡터화하는데, 기존 단어에 <, 와 >를 붙인 \<apple> 토큰입니다.

FastText의 인공 신경망을 학습한 후에는 데이터 셋의 모든 단어의 각 n-gram에 대해서 워드 임베딩이 됩니다. 이렇게 되면 장점은 데이터 셋만 충분한다면 위와 같은 내부 단어(Subword)를 통해 모르는 단어(Out Of Vocabulary, OOV)에 대해서도 다른 단어와의 유사도를 계산할 수 있다는 점입니다. 가령, FastText에서 birthplace(출생지)란 단어를 학습하지 않은 상태라고 해봅시다. 하지만 다른 단어에서 birth와 place라는 내부 단어가 있었다면, FastText는 birthplace의 벡터를 얻을 수 있습니다. 이는 모르는 단어에 제대로 대처할 수 없는 Word2Vec, GloVe와는 다른 점입니다.

코드 구현을 통해 FastText와 Word2Vec을 비교해봅시다.

{% code lineNumbers="true" %}
```python
import re
import urllib.request
import zipfile
from lxml import etree
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.models import FastText, Word2Vec, KeyedVectors

# 데이터 다운로드
urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/09.%20Word%20Embedding/dataset/ted_en-20160408.xml", filename="ted_en-20160408.xml")

targetXML = open('ted_en-20160408.xml', 'r', encoding='UTF8')
target_text = etree.parse(targetXML)

# xml 파일로부터 <content>와 </content> 사이의 내용만 가져온다.
parse_text = '\n'.join(target_text.xpath('//content/text()'))

# 정규 표현식의 sub 모듈을 통해 content 중간에 등장하는 (Audio), (Laughter) 등의 배경음 부분을 제거.
# 해당 코드는 괄호로 구성된 내용을 제거.
content_text = re.sub(r'\([^)]*\)', '', parse_text)

# 입력 코퍼스에 대해서 NLTK를 이용하여 문장 토큰화를 수행.
sent_text = sent_tokenize(content_text)

# 각 문장에 대해서 구두점을 제거하고, 대문자를 소문자로 변환.
normalized_text = []
for string in sent_text:
     tokens = re.sub(r"[^a-z0-9]+", " ", string.lower())
     normalized_text.append(tokens)

# 각 문장에 대해서 NLTK를 이용하여 단어 토큰화를 수행.
result = [word_tokenize(sentence) for sentence in normalized_text]

# Word2Vec과 FastText의 비교
model_wv = Word2Vec(sentences=result, vector_size=100, window=5, min_count=5, workers=4, sg=0)
model_ft = FastText(result, vector_size=100, window=5, min_count=5, workers=4, sg=1)
```
{% endcode %}

{% code lineNumbers="true" %}
```python
model_wv.wv.most_similar("electrofishing")
```
{% endcode %}

```
KeyError: "Key 'electrofishing' not present in vocabulary"
```

{% code lineNumbers="true" %}
```python
model_ft.wv.most_similar("electrofishing")
```
{% endcode %}

```
[('electrolyte', 0.8595626950263977),
 ('electrolux', 0.8584996461868286),
 ('electroshock', 0.8453683853149414),
 ('electro', 0.8443098068237305),
 ('electrochemical', 0.8417826294898987),
 ('electroencephalogram', 0.8407151699066162),
 ('electrogram', 0.8311992883682251),
 ('airbus', 0.82285076379776),
 ('airbag', 0.8175927996635437),
 ('electrons', 0.8162853717803955)]
```
