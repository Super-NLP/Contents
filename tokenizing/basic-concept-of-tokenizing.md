---
description: 토크나이징의 기본 개념에 대해 알아보자.
---

# 🦂 Basic Concept of Tokenizing

## Ⅰ. Character Tokenizing

자연어 처리는 일반적으로 토큰화, 단어 집합(=vocabulary) 생성, 정수 인코딩, 패딩, 벡터화의 과정을 거친다. 주어진 텍스트를 단어 또는 문자 단위로 자르는 것을 토큰화(=Tokenizing)라고 한다. 여기서는 토큰과 토큰나이징의 개념에 대해 중점적으로 다루도록 하겠다. 기초적인 개념은 다음과 같다.

* **토큰 (Token)**
  * 문장 형태의 데이터에서 의미가 있는 가장 작은 정보 단위
* **토크나이징 (Tokenizing)**
  * 문장 형태의 데이터를 토큰 단위로 나누는 과정
* **토크나이저 (Tokenizer)**
  * 문장 형태의 데이터를 토큰 단위로 나눠주는(=토크나이징을 수행하는) 객체

중요한것은 코퍼스를 모델이 잘 이해할 수 있도록 최소 의미를 담게끔 분절하는 작업을 말하며 이에 대한 방법도 굉장히 많다. 사실 가장 간단한 토크나이징은 가벼운 python code만 이용해도 수행할 수 있다.

{% code lineNumbers="true" %}
```python
# Simple tokenizing
text = "Tokenizing text is a core task of NLP"
tokenized_text = list(text)
tokenized_text
```
{% endcode %}

```
['T', 'o', 'k', 'e', 'n', 'i', 'z', 'i', 'n', 'g', ' ', 't', 'e', 'x', 't', ' ', 'i', 's', ' ', 'a', ' ', 'c', 'o', 'r', 'e', ' ', 't', 'a', 's', 'k', ' ', 'o', 'f', ' ', 'N', 'L', 'P']
```

별거 아닌것 같지만 이것도 하나의 토크나이징이다. \
문자 수준의 토크나이징(=Character Level Tokenizing)이라고 부른다.

다만 이러한 문자수준의 토크나이징은 거의 사용하지 않는다. 왜냐하면 단어같은 언어 구조를 데이터에서 학습해야하는 큰 제약이 있고 이를 위해 상당량의 계산, 메모리, 데이터가 필요해 비효율적이기 때문이다.  방식은 다르지만 토크나이징이 되었다는 가정하에 vocab을 만들고 이를 이용해 어떻게 코퍼스를 컴퓨터가 이해하기 편하게 바꾸는지 소개하겠다.

{% code lineNumbers="true" %}
```python
# Numericalization : 수치화
# Set Vocab
vocab = {character : idx for idx, character in enumerate(sorted(set(tokenized_text)))}
print(vocab)
```
{% endcode %}

```
{' ': 0, 'L': 1, 'N': 2, 'P': 3, 'T': 4, 'a': 5, 'c': 6, 'e': 7, 'f': 8, 'g': 9, 'i': 10, 'k': 11, 'n': 12, 'o': 13, 'r': 14, 's': 15, 't': 16, 'x': 17, 'z': 18}
```

물론 Vocab을 만드는데 토크나이징 방식이 영향을 준다. 방식에 변화에 따라 만들어지는 Vocab 내용이 달라질 수 있음을 유의하고 지금은 큰 맥락만 보도록 하자. Vocab이 만들어졌으면 이것을 가지고 문장을 숫자로, 숫자를 문장으로 변화시킬 수 있다.

{% code lineNumbers="true" %}
```python
# Text to Index : Encoding
input_ids = [vocab[token] for token in tokenized_text]
print(input_ids)
```
{% endcode %}

```
[4, 13, 11, 7, 12, 10, 18, 10, 12, 9, 0, 16, 7, 17, 16, 0, 10, 15, 0, 5, 0, 6, 13, 14, 7, 0, 16, 5, 15, 11, 0, 13, 8, 0, 2, 1, 3]
```

{% code lineNumbers="true" %}
```python
# Index to Text : Decoding
reverse_vocab = {value: key for key, value in vocab.items()}
output_tokens = [reverse_vocab[val] for val in input_ids]
print(output_tokens)
```
{% endcode %}

```
['T', 'o', 'k', 'e', 'n', 'i', 'z', 'i', 'n', 'g', ' ', 't', 'e', 'x', 't', ' ', 'i', 's', ' ', 'a', ' ', 'c', 'o', 'r', 'e', ' ', 't', 'a', 's', 'k', ' ', 'o', 'f', ' ', 'N', 'L', 'P']
```

각각의 과정을 Encoding, Decoding이라고 한다. 실제로 언어모델링을 할때 'Embedding'이라는 과정을 거치게 되는데 이것을 거치기 위해선 문자를 숫자형태로 바꿔주어야 하고 이것을 Vectorizing(=벡터화)이라고 부른다.&#x20;

## Ⅱ. Word Tokenizing

현재 보편적으로 사용되는 토크나이징 방식은 Word Tokenizing을 기반으로 한다고 봐야할 것이다. 여기부터는 사용하는 언어가 영어냐 한국어냐에 따라 라이브러리의 선택과 고려해야할 사항도 달라진다. 언어별 대표적인 토크나이저 라이브러리는 다음과 같다.

* **Korean**
  * KoNLPy
    * Mecab
    * Hannanum
    * Kkma
    * Komoran
    * Okt
* **English**
  * spaCy
  * nltk
* **Japanese**
  * Mecab
* **Chinese**
  * Standford Parser
  * PKU Parser
  * Jieba

{% code title="한국어 토크나이징 패키지의 수행시간 비교" lineNumbers="true" %}
```python
# Compare to Tokenizer Library
import time
import konlpy
from konlpy.tag import Okt, Kkma, Hannanum, Komoran, Mecab

def tokenize(text, object):
    start = time.time()
    if object == 'okt':
        okt = Okt()
        print(okt.morphs(text))
        end = time.time()
        print(f"> Okt() Execute Time : {end - start}")
    elif object == 'kkma':
        kkma = Kkma()
        print(kkma.morphs(text))
        end = time.time()
        print(f"> Kkma() Execute Time : {end - start}")
    elif object == 'hannanum':
        hannanum = Hannanum()
        print(hannanum.morphs(text))
        end = time.time()
        print(f"> Hannanum() Execute Time : {end - start}")
    elif object == 'komoran':
        komoran = Komoran()
        print(komoran.morphs(text))
        end = time.time()
        print(f"> Komoran() Execute Time : {end - start}")
    elif object == 'mecab':
        mecab = Mecab()
        print(mecab.morphs(text))
        end = time.time()
        print(f"> Mecab() Execute Time : {end - start}")


text = "Tokenizing text is a core task of NLP"

tokenize(text, 'okt')
tokenize(text, 'kkma')
tokenize(text, 'hannanum')
tokenize(text, 'komoran')
tokenize(text, 'mecab')

```
{% endcode %}

```
['Tokenizing', 'text', 'is', 'a', 'core', 'task', 'of', 'NLP']
> Okt() Execute Time : 0.03143143653869629
['Tokenizing', 'text', 'is', 'a', 'core', 'task', 'of', 'NLP']
> Kkma() Execute Time : 0.012807130813598633
['Tokenizing', 'text', 'is', 'a', 'core', 'task', 'of', 'NLP']
> Hannanum() Execute Time : 2.0569846630096436
['Tokenizing', 'text', 'is', 'a', 'core', 'task', 'of', 'NLP']
> Komoran() Execute Time : 3.9626095294952393
['Tokenizing', 'text', 'is', 'a', 'core', 'task', 'of', 'NLP']
> Mecab() Execute Time : 0.002543926239013672
```

한국어 토크나이저의 경우 **Mecab()**이 가장 빠른걸 알 수 있다!&#x20;

더 자세한 단어 토크나이징 방식은 다음장에서 다루도록 하겠다.
