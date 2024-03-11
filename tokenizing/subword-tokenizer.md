---
description: Subword Tokenizer에 대해 알아봅시다.
---

# 🦂 Subword Tokenizer

여기부터는 **Word Tokenizer**에 대해서 알아보겠습니다.

{% code lineNumbers="true" %}
```python
text = "Tokenizing text is a core task of NLP."
tokenized_text = text.split()
print(tokenized_text)
```
{% endcode %}

```
['Tokenizing', 'text', 'is', 'a', 'core', 'task', 'of', 'NLP.']
```

위의 결과처럼 가장 간단한 Word Tokenizer는 공백을 이용해 텍스트를 토큰화하는 것입니다. 간단한 Python Code로 구현할 수 있죠. 그 이후에는 앞서 보였던 Vocab 형성과 Encode, Decode 과정을 동일하게 겪게 됩니다. \
하지만 이 방식도 드러나지 않은 문제가 있는데 바로 구두점이 고려되지 않는다는 것입니다. `"NLP."`과 같이 구두점과 단어가 하나의 토큰으로 처리되는것을 볼 수 있습니다.&#x20;

## Ⅰ. Frequency based Tokenizing

이렇게 tokenizing한다면 형성되는 vocab도 굉장히 크기가 커질 것 입니다. 이는 곧 신경망의 파라미터수가 늘어나고 비용과 메모리 가용면에서 굉장히 비효율적이죠. 그래서 보통 vocab의 크기를 제한하는 방법으로 드물게 등장하는 단어를 무시하는 것입니다. **즉, 빈도수에 기반하여 tokenizing하고 vocab을 만드는 것**이죠. 이렇게 한다면 vocab에 없는 단어는 unknown 으로 분류해서 UNK 토큰으로 매핑되고 일부분 정보손실되는것을 막을 수 없게 됩니다.

{% code lineNumbers="true" %}
```python
from konlpy.tag import Mecab
import re

corpus = """자연 언어 처리에서 말하는 형태소 분석이란 어떤 대상 어절을 최소의 의미 단위인 '형태소'로 분석하는 것을 의미한다. (형태소는 단어 그 자체가 될 수도 있고, 일반적으로는 단어보다 작은 단위이다.) 정보 검색 엔진에서 한국어의 색인어 추출에 많이 사용한다. 
형태소 분석 단계에서 문제가 되는 부분은 미등록어, 오탈자, 띄어쓰기 오류 등에 의한 형태소 분석의 오류, 중의성이나 신조어 처리 등이 있는데, 이들은 형태소 분석에 치명적인 약점이라 할 수 있다. 복합 명사 분해도 형태소 분석의 어려운 문제 중 하나이다. 
복합 명사란 하나 이상의 단어가 합쳐서 새로운 의미를 생성해 낸 단어로 '봄바람' 정보검색' '종합정보시스템' 등을 그 예로 들 수 있다. 
이러한 단어는 한국어에서 띄어쓰기에 따른 형식도 불분명할 뿐만 아니라 다양한 복합 유형 등에 따라 의미의 통합이나 분해가 다양한 양상을 보이기 때문에 이들 형태소를 분석하는 것은 매우 어려운 문제이다. 
기계적으로 복합명사를 처리하는 방식 중의 하나는, 음절 단위를 기반으로 하는 바이그램이 있다. 
예를 들어, '복합 명사'는 음절 단위로 '복합+명사', '복+합명사', '복합명+사'의 세 가지 형태로 쪼갤 수 있고, 이 중 가장 적합한 분해 결과를 문서 내에서 출현하는 빈도 등의 추가 정보를 통해 선택하는 알고리즘이 있을 수 있다. 
일반적으로, 다양하게 쪼개지는 분석 결과들 중에서 적합한 결과를 선택하기 위해, 테이블 파싱이라는 동적 프로그래밍 방법을 사용한다."""

# 한국어 형태소 분석기 객체 생성
m = Mecab()

# 정규표현식을 이용한 코퍼스 정제
pattern  = '[^a-zA-Z가-힣\d\n]'
re_corpus = re.sub(pattern=pattern, repl=' ', string=corpus)
re_corpus = re.sub(pattern='\s{1,}', repl=' ', string=re_corpus)

# Mecab을 이용한 한국어 형태소 분석과 토크나이징
analysis_text = m.morphs(re_corpus)

# 토크나이징 후 분절된 단어의 출현빈도가 1인 단어를 필터링하는 작업 수행
target_list = list(set(analysis_text))
before_cnt = len(target_list) # 이전 단어 개수
for item in target_list:
    if analysis_text.count(item) < 2:
        target_list.remove(item)
after_cnt = len(target_list) # 이후 단어 개수

# 결과 확인
print(f"Before count : {before_cnt} / After count : {after_cnt}")
```
{% endcode %}

```
Before count : 141 / After count : 89
```

위의 결과처럼 기존에 주어진 Corpus가 Mecab 형태소 분석기에 의해 141개로 분절되었고 출현 빈도수가 1인 단어들을 다 지워주니 최종적으로 89개의 단어가 남은것을 확인할 수 있습니다. 만약 여기서 줄어든 52개의 단어가 포함된 문장을 encode/decode 시킨다면 해당 단어가 UNK 토큰으로 대체되어 나타날 것입니다.

## Ⅱ. BPE (Byte Pair Encoding)

반복적으로 언급했듯 우리의 데이터는 무한하지 않기때문에 UNK 토큰을 통해 모르는 의미의 토큰을 처리할 수 밖에 없습니다. 이러한 문제는 **OOV(Out of Vocabulary)**라고 부릅니다. 이러한 OOV 문제를 해결하기위해 나타난 방법론에 대해 알아봅시다.

BPE 알고리즘은 Byte Pair Encoding의 약자로 1994년에 데이터 압축을 위해 생긴 알고리즘입니다. 가장 많이 등장하는 바이트 쌍(Byte Pair)을 새로운 단어로 치환하여 압축하는 작업을 반복적으로 수행합니다. 이렇게 하면 접두어와 접미어의 의미를 잘 파악하고 처음 등장하는 단어는 문자들의 조합으로 나타내어 OOV문제를 해결할 수 있다는 개념으로 사용되었습니다.

{% code title="예제 : https://arxiv.org/pdf/1508.07909.pdf" lineNumbers="true" %}
```python
import re, collections

# 임의의 데이터에 포함된 단어들
# 오른쪽 정수는 데이터에 해당 단어가 포함된 빈도수
vocab = {
    'l o w '       : 5,
    'l o w e r '   : 2,
    'n e w e s t ' : 6,
    'w i d e s t ' : 3
}

num_merges = 5

def get_stats(vocab):
  pairs = collections.defaultdict(int)
  
  for word, freq in vocab.items():
    symbols = word.split()
    
    for i in range(len(symbols) -1):
      pairs[symbols[i], symbols[i+1]] += freq
    
  return pairs

def merge_vocab(pair, v_in):
  v_out = {}
  bigram = re.escape(' '.join(pair))
  p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
  
  for word in v_in:
    w_out = p.sub(''.join(pair), word)
    v_out[w_out] = v_in[word]
  
  return v_out, pair[0] + pair[1]

token_vocab = []

for i in range(num_merges):
  print(">> step {0}".format(i+1))
  
  pairs = get_stats(vocab)
  best = max(pairs, key=pairs.get)
  vocab, merge_tok = merge_vocab(best, vocab)
  print('다음 문자 쌍을 치환 : ', merge_tok)
  print('변환된 Voab : \n', vocab, '\n')
  
  token_vocab.append(merge_tok)

print('Merge Vocab: ', token_vocab)
```
{% endcode %}

```
>> step 1
다음 문자 쌍을 치환 :  es
변환된 Voab : 
 {'l o w ': 5, 'l o w e r ': 2, 'n e w es t ': 6, 'w i d es t ': 3} 

>> step 2
다음 문자 쌍을 치환 :  est
변환된 Voab : 
 {'l o w ': 5, 'l o w e r ': 2, 'n e w est ': 6, 'w i d est ': 3} 

>> step 3
다음 문자 쌍을 치환 :  lo
변환된 Voab : 
 {'lo w ': 5, 'lo w e r ': 2, 'n e w est ': 6, 'w i d est ': 3} 

>> step 4
다음 문자 쌍을 치환 :  low
변환된 Voab : 
 {'low ': 5, 'low e r ': 2, 'n e w est ': 6, 'w i d est ': 3} 

>> step 5
다음 문자 쌍을 치환 :  ne
변환된 Voab : 
 {'low ': 5, 'low e r ': 2, 'ne w est ': 6, 'w i d est ': 3} 

Merge Vocab:  ['es', 'est', 'lo', 'low', 'ne']
```

위와 같은 방식대로 동작하는데 기존의 vocab에는 "low, lower, newest, widest"라는 단어만 존재했지만 BPE를 적용하면 lowest라는 단어가 들어와도 어느정도 의미가 파악된 low와 est의 결합으로 의미를 이해할 수 있습니다. 그러나 이러한 방식도 완벽한 것은 아니기에 Google에서 이것을 변형하여 WPM라는 새로운 방식을 제안합니다.

## Ⅲ. WPM (WordPiece Model)

WPE (WordPiece Model)은 이전의 BPE와 달리 아래 2가지 특징을 지닙니다.

* 공백에 대한 복원을 위해 단어의 시작 부분에 언더바('\_')를 추가합니다.
* 기존의 빈도수 기반이 아닌 우도(Likelihood)를 증가시키는 방향으로 문자 쌍을 합칩니다.

다음 수식을 보고 이해를 해봅시다. 다음은 병합후보가 $$a, b$$일 때 판단의 근거가 되는 값을 계산하는 방법입니다. 여기서 사용되는 $$\#a, \#b, \#ab$$는 각각 $$a, b, ab$$라는 문자열의 빈도수, $$n$$은 전체 글자수를 가리킵니다. 즉, 분자는 $$ab$$가 연달아서 등장할 확률을 의미하고 분모는 $$a, b$$가 각각 등장할 확률의 곱을 의미하게 됩니다.

$$
{ { \#ab \over n }  \over {\#a \over n} \times {\#b \over n}}
$$

이 수식의 값이 커지려면 $$a$$ 와 $$b$$가 서로 독립이라는 가정하에 이 두개가 자주 동시에 등장해야 합니다. 즉, WordPiece에서는 병합후보에 오른 쌍을 미리 병합해보고 잃는 것과 가치등을 판단한 후 병합합니다. 워드피스는 병합 대상 전체 후보들 가운데 위와 같이 계산한 값이 가장 높은 쌍을 합칩니다.

워드피스는 어휘 집합(`vocab.txt`)만 가지고 토큰화합니다. 워드피스에서는 분석 대상 어절에 어휘 집합에 있는 서브워드가 포함돼 있을 때 해당 서브워드를 어절에서 분리합니다. 단, 이러한 서브워드 후보가 여럿 있을 경우 가장 긴 서브워드를 선택합니다. 이후 어절의 나머지에서 어휘 집합에 있는 서브워드를 다시 찾고(최장 일치 기준), 또 분리합니다. 분석 대상 문자열에서 서브워드 후보가 하나도 없으면 해당 문자열 전체를 미등록 단어로 취급합니다.

{% code lineNumbers="true" %}
```python
from transformers import AutoTokenizer, DistilBertTokenizer

# checkpoint 이름을 이용한 자동 클래스 "AutoTokenizer"
model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

# 특정 클래스를 수동으로 로드
# distilbert_tokenizer = DistilBertTokenizer.from_pretrained(model_ckpt)

test = "Tokenizing text is a core task of NLP."
encoded_text = tokenizer(test)
print(f"- Token → ids : {encoded_text}")
tokens = tokenizer.convert_ids_to_tokens(encoded_text['input_ids'])
print(f"- Ids → Token : {tokens}")
print(f"- Original Str : {tokenizer.convert_tokens_to_string(tokens)}")
print(f"- Vocab Size : {tokenizer.vocab_size}")
```
{% endcode %}

```
- Token → ids : {'input_ids': [101, 19204, 6026, 3793, 2003, 1037, 4563, 4708, 1997, 17953, 2361, 1012, 102], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
- Ids → Token : ['[CLS]', 'token', '##izing', 'text', 'is', 'a', 'core', 'task', 'of', 'nl', '##p', '.', '[SEP]']
- Original Str : [CLS] tokenizing text is a core task of nlp. [SEP]
- Vocab Size : 30522
```

이번에는 임시 데이터셋을 가지고 실제 BERT 토크나이저를 구축해봅시다.

{% code lineNumbers="true" %}
```python
from Korpora import Korpora
from tokenizers import BertWordPieceTokenizer
import os

# Korpora를 통해 nsmc 데이터셋을 가져옵니다.
nsmc = Korpora.load("nsmc", force_download=True)
def write_lines(path, lines):
    with open(path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(f'{line}\n')
write_lines("/content/train.txt", nsmc.train.get_all_texts())
write_lines("/content/test.txt", nsmc.test.get_all_texts())

# BERT 토크나이저를 하나 만들고 다운받은 데이터셋으로 학습시켜줍니다.
wordpiece_tokenizer = BertWordPieceTokenizer(lowercase=False)
wordpiece_tokenizer.train(
    files=["/content/train.txt", "/content/train.txt"],
    vocab_size=10000,
)

# 학습시킨 BERT 토크나이저를 저장합니다.
os.makedirs("/content/wordpiece", exist_ok=True)
wordpiece_tokenizer.save_model("/content/wordpiece")
```
{% endcode %}

```
    Korpora 는 다른 분들이 연구 목적으로 공유해주신 말뭉치들을
    손쉽게 다운로드, 사용할 수 있는 기능만을 제공합니다.

    말뭉치들을 공유해 주신 분들에게 감사드리며, 각 말뭉치 별 설명과 라이센스를 공유 드립니다.
    해당 말뭉치에 대해 자세히 알고 싶으신 분은 아래의 description 을 참고,
    해당 말뭉치를 연구/상용의 목적으로 이용하실 때에는 아래의 라이센스를 참고해 주시기 바랍니다.

    # Description
    Author : e9t@github
    Repository : https://github.com/e9t/nsmc
    References : www.lucypark.kr/docs/2015-pyconkr/#39

    Naver sentiment movie corpus v1.0
    This is a movie review dataset in the Korean language.
    Reviews were scraped from Naver Movies.

    The dataset construction is based on the method noted in
    [Large movie review dataset][^1] from Maas et al., 2011.

    [^1]: http://ai.stanford.edu/~amaas/data/sentiment/

    # License
    CC0 1.0 Universal (CC0 1.0) Public Domain Dedication
    Details in https://creativecommons.org/publicdomain/zero/1.0/

[nsmc] download ratings_train.txt: 14.6MB [00:00, 64.9MB/s]                            
[nsmc] download ratings_test.txt: 4.90MB [00:00, 29.7MB/s]                           
['/content/wordpiece/vocab.txt']
```

위와 같이 실행되고 만들어진 vocab.txt를 열어보면 아래와 같이 나타납니다.

```
[PAD]
[UNK]
[CLS]
[SEP]
[MASK]
!
"
%
&
'
(
)
*
+
,
-
.
/

...

장나라
저러
전국
전락
정리
정재
조정
직장
찾을
총격
```

## Ⅳ. SentencePiece

SentencePiece도 구글에서 제작되어 공개되었습니다. 우리가 이제까지 해왔던 과정인 "코퍼스 정제 → 토크나이징"의 과정을 한번에 할 수 있다면 어떨까요? 즉, 사전토큰화작업없이 바로 데이터에서 단어분리 토크나이징을 할 수 있다면 꽤나 편리하겠죠? Google의 SentencePiece는 이점을 살려서 구현되었고 언어의 종속을 받지 않습니다.

이에 대한 깃허브 링크를 남겨놓겠습니다. [https://github.com/google/sentencepiece](https://github.com/google/sentencepiece)



## Ⅴ. HuggingFace Tokenizer

최근들어서는 huggingface를 사용하지 않는 사람이 없을 것입니다. 자연어처리 스타트업인 허깅페이스에서 개발한 tokenizers 라이브러리는 다양한 서브워드 토크나이저를 제공합니다. 여기서도 WPM때와 같이 BERT의 Tokenizer를 불러와 사용해보겠습니다.

{% code lineNumbers="true" %}
```python
import pandas as pd
import urllib.request
from tokenizers import BertWordPieceTokenizer

# 데이터 로드 및 정제
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt")
naver_df = pd.read_table('ratings.txt')
naver_df = naver_df.dropna(how='any')
with open('naver_review.txt', 'w', encoding='utf8') as f:
    f.write('\n'.join(naver_df['document']))

# 토크나이저 선언
tokenizer = BertWordPieceTokenizer(lowercase=False)

# 학습 매개변수 설정
data_file = 'naver_review.txt'
vocab_size = 30000
limit_alphabet = 6000
min_frequency = 5

# 토크나이저 학습
tokenizer.train(files=data_file,
                vocab_size=vocab_size,
                limit_alphabet=limit_alphabet,
                min_frequency=min_frequency)

# 만들어진 vocab 저장
tokenizer.save_model('./')

# 만들어진 vocab 로드
df = pd.read_fwf('vocab.txt', header=None)
print(df)
```
{% endcode %}

<figure><img src="../.gitbook/assets/image (4) (1) (1).png" alt="" width="163"><figcaption><p>df</p></figcaption></figure>

```python
encoded = tokenizer.encode('아 배고픈데 짜장면먹고싶다')
print('토큰화 결과 :',encoded.tokens)
print('정수 인코딩 :',encoded.ids)
print('디코딩 :',tokenizer.decode(encoded.ids))
```

```
토큰화 결과 : ['아', '배고', '##픈', '##데', '짜장면', '##먹고', '##싶다']
정수 인코딩 : [2111, 20630, 3860, 3268, 24680, 7872, 7379]
디코딩 : 아 배고픈데 짜장면먹고싶다
```

위와 같은 결과를 볼 수 있습니다. 그 밖에도 다양한 토크나이저들을 지원하는데 종류는 아래와 같습니다.

* **BertWordPieceTokenizer** : BERT에서 사용된 워드피스 토크나이저(WordPiece Tokenizer)
* **CharBPETokenizer** : 오리지널 BPE
* **ByteLevelBPETokenizer** : BPE의 바이트 레벨 버전
* **SentencePieceBPETokenizer** : 앞서 본 패키지 센텐스피스(SentencePiece)와 호환되는 BPE 구현체

```python
from tokenizers import ByteLevelBPETokenizer, CharBPETokenizer, SentencePieceBPETokenizer

tokenizer = SentencePieceBPETokenizer()
tokenizer.train('naver_review.txt', vocab_size=10000, min_frequency=5)

encoded = tokenizer.encode("이 영화는 정말 재미있습니다.")
print(encoded.tokens)
```

```
['▁이', '▁영화는', '▁정말', '▁재미있', '습니다.']
```
