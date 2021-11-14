# CartoonizedGanExport

`cycle gan` 을 기반으로 학습한 모델을 (`현실 -> 흑백 만화체`) 대상으로
원본 레포가 너무나 많아, 필요한 부분만 추출하고,  
그 중에서도 `torch jit` 를 통해서 필요로한 부분만 추출하는 방법을 정리해놓은 Repo 입니다.

### Require Environment
아래 라이브러리 정도만 필요로하니, 쉽게 환경을 구축하실 수 있으실 겁니다.  
굳이 아래 설정과 버젼까지 동일할 필요는 없으며, 이케저케 해서 실행을 하실 수 있으실 겁니다.

```
Pillow==7.1.2
torch==1.10.0
torchvision==0.11.1
```



|원본|CartoonizeGan 적용후|
|------|---|
|<img src="./sample/a.jpeg" width="300" height="300">|<img src="./sample/cg_a.jpeg" width="300" height="300">|
