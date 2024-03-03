---
description: 텍스트 데이터를 수집하거나 오픈소스로 공개되어있는 데이터에 대하여 알아봅시다.
---

# 🐌 Data Crawling

자연어처리를 하기위해선 먼저 데이터 수집을 진행해야 한다. 데이터를 구하는 방식은 매우 다양하다. 공개된 데이터를 사용하거나 (Public Data), 구매하여 사용할 수 있고, 논문을 위한 데이터에서 발췌하여 적용할 수도 있다. 또한 크롤링을 통한 수집을 진행할수도 있다. 단, 웹사이트에서 무작정 데이터를 크롤링할 경우 법적인 문제로 이어질 수 있기때문에 크롤링 여부를 먼저 확인하는것이 중요하다. (크롤링에 관해서는 해당 사이트의 `robots.txt`를 확인해보아라. )

{% code title="TED의 robots.txt" %}
```
$ wget https://www.ted.com/robots.txt
$ cat robots.txt
User-agent: *
Disallow: /latest
Disallow: /latest-talk
Disallow: /latest-playlist
Disallow: /people
Disallow: /profiles
Disallow: /conversations

User-agent: Baiduspider
Disallow: /search
Disallow: /latest
Disallow: /latest-talk
Disallow: /latest-playlist
Disallow: /people
Disallow: /profiles
```
{% endcode %}

일반적으로 크롤링은 Python의 **Selenium**이나 **Beautiful Soup**을 사용한다.
