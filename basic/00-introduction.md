---
description: 2025.04.11.
cover: ../.gitbook/assets/logo_test.png
coverY: 0
layout:
  cover:
    visible: true
    size: hero
  title:
    visible: true
  description:
    visible: true
  tableOfContents:
    visible: true
  outline:
    visible: true
  pagination:
    visible: true
---

# 00 Introduction

요즘 커리어에 대한 고민이 깊다. 스스로도 여러방면의 고민을 하게되는 것 같다. 이를테면 **'회사에서 하는 일과 개인실력의 향상의 관계성'**, **'내가 하는 분야의 성장할 수 있는 환경'**  등등 많은 것 같다. 고민이 되는 건 어찌보면 당연한 것이지만 당장 해결할 수 있는 문제는 생각보다 많지 않고 방향성이 더 중요하다는 생각을 하게된다. 최근에 팀 동료의 한 멘트가 머리를 크게 울렸었는데, 그의 말을 의역해서 옮기자면 <mark style="color:red;">**"본인은 팀 동료들이 Transformer 아키텍쳐를 계속해서 반복해 보았으면 좋겠다."**</mark>는 말을 하셨다. 이 말을 듣고 곰곰히 생각해보니 필도 옛날에는 줄곧 기본을 돌아보는 시간을 가졌었는데 요즘엔 발전을 따라가는데 급급하다보니 여러가지 기본적인 것을 되짚을 여력이 없어진 것 같다. 그러니 컴팩트하게 기본을 돌아보는 시간을 마련하고 그에 따라 정리를 하려고 한다.&#x20;

당연히 나의 <mark style="color:green;">**주 분야는 RAG/Agent**</mark>이다. Encoder base의 Model이라면 또 모를까 Decoder base Model은 우선순위에 밀리기는 한다. 그렇지만 가장 기본적인 뼈대는 LLM이고 방금 말한 내용을 감안하더라도 스스로의 Level-Up을 생각하면 필수적이라고 생각한다. 언젠가 해야지, 해야지 하던 기본을 되짚는 시간을 <mark style="background-color:red;">**팀 동료의 말 한마디로부터 불이 지펴져서 시작하게 되었다. 그에게 감사인사를 남긴다.**</mark>

어떠한 방식대로 진행할 것인가 하면 **허정준**님의 **"LLM을 활용한 실전 AI 애플리케이션 개발"** 도서를 활용하여 기본을 되짚을 예정이고, Chapter별 학습을 진행한 뒤 해당 Chapter에서 리뷰할 내용을 추려 간략하게 Review하며 블로그 포스팅을 남기는 것으로 진행하려고 한다.

***

아래에 Index를 달아놓을테니 참고바란다😄

* **1부 LLM의 기초 뼈대 세우기**
  * [LLM 지도](01-1-llm.md)
  * LLM의 중추, 트랜스포머 아키텍쳐 살펴보기
  * 트랜스포머 모델을 다루기 위한 허깅페이스 트랜스포머 라이브러리
  * 말 잘듣는 모델 만들기
* **2부 LLM 길들이기**
  * GPU 효율적인 학습
  * sLLM 학습하기
  * 모델 가볍게 만들기
  * sLLM 서빙하기
* **3부 LLM을 활용한 실전 애플리케이션 개발**
  * LLM 애플리케이션 개발하기
  * 임베딩 모델로 데이터 의미 압축하기
  * 자신의 데이터에 맞춘 임베딩 모델 만들기 : RAG 개선하기
  * 벡터 데이터베이스로 확장하기 : RAG 구현하기
  * LLM 운영하기
* **4부 멀티 모달, 에이전트 그리고 LLM의 미래**
  * 멀티 모달 LLM
  * LLM 에이전트
  * 새로운 아키텍쳐
