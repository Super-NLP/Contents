---
description: 언어모델과 LLM을 포함하여 어떻게 서비스로 연결시켜 서빙하는지에 대해 알아보자.
coverY: 0
---

# Introduce

> **Keyword**\
> \-> Docker, Kubernates(k8s), FastAPI, MLflow, vLLM, Lakehouse, Blue/Green, LoadBalancing

### **✅ How to serve LLM in real service**

* **Prototype** or **PoC Level**
  * 서빙도 FastAPI + Uvicorn 활용해서 간단하게 수행 가능
* **Product Level**
  * 공통적으로 Service (Server) ↔ Model (Proxy Server + Docker + Kubernates가 정석으로 활용되는듯 함. PyTorch, Tensorflow, Langchain, Ollama, Azure Databricks, MLFlow, Lakehouse, etc… 여러가지 배포환경에 따라 사용하는 프레임워크는 다양하게 나뉠 것으로 보임.
  * **On-premise 환경**
    * 온프레미스 환경은 원격환경에서 서버를 운영하는 클라우드와는 대비되는 개념으로 **기업이 서버를 자체적으로 보유하고 직접 설치 및 운영하는 방식**을 의미함.
    * 데이터 보안이나 외부망 활용이 제한되는 경우, 필수적으로 LangChain 프레임워크를 활용해야할 것으로 판단됨
  * **Cloud 환경**
    * Azure 클라우드 기반으로 **Azure Databricks** 등의 다양한 기능을 활용하여 LangChain 어플리케이션을 개발할 수 있음.



### 📜 Reference

[LangChain: LLM 서비스를 어떻게 개발할 수 있을까?](https://bigwaveai.tistory.com/76)

[새로운 루다를 지탱하는 모델 서빙 아키텍처 — 1편: A/B 테스트를 위한 구조 설계](https://tech.scatterlab.co.kr/serving-architecture-1/)

[새로운 루다를 지탱하는 모델 서빙 아키텍처 — 2편: ArgoCD와 모델 서빙](https://tech.scatterlab.co.kr/serving-architecture-2/)

[HyperCLOVA 서빙 프레임워크 선정 | CLOVA Engineering Blog](https://engineering.clova.ai/posts/2022/01/hyperclova-part-1)

[NSML - 분산 학습 플랫폼의 스케줄링 요구 사항과 해결 방안 | CLOVA Engineering Blog](https://engineering.clova.ai/posts/2022/08/nsml-scheduler-part-1)

[vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html)
