## 논문 구현 코드 

### 개요 
- 제목: Prompt-Guided Dataset Generation to Mitigate Biases in a Language Model
- 목표: 생성형 언어 모델을 이용하여 기존 언어 모델에서 나타나는 편향을 해소할 수 있는 데이터셋 생성
- 과정
  - 기존 언어 모델 (GPT-2) 에서 특성에 따른 편향 확인
    - 범주 개수가 가장 적은 이진 성별을 특징으로 설정 
    - 특성만 다르고 맥락은 같은 두 개의 미완성 문장을 입력
    - 그 다음에 올 단어의 확률 확인 - vocabulary 내의 각 token에 부여되는 확률 확인
  - 편향을 상쇄하는 데이터셋 생성
    - token별 확률 및 품사 정보을 바탕으로 편향 정보와 편향 상쇄 정보 추출 
    - 준비한 포맷에 token을 바꿔 넣으면서 자연어 프롬프트 생성 
    - OpenAI API(GPT-3.5-turbo)에 프롬프트 입력 후 재학습에 사용할 문장 생성 
  - 재학습 전후 벤치마크로 측정된 편향 정도의 차이 확인
    - StereoSet benchmark 이용 
- 결과
  - 모든 실험 설정(token filtering parameter setting)에서 학습 전보다 편향 완화
  - 다른 방법론과 비교했을 때, 상대적으로 작은 재학습 데이터셋으로도 유사한 효과를 확인
---
#### 폴더 구조 
  - `preprocess`: 기존 모델의 편향 확인을 위한 데이터 전처리 및 프롬프트 생성
  - `bias-detect`: 기존 모델의 편향 확인
  - `generate`: OpenAI API 이용해서 데이터셋 생성 
  - `train`: 생성한 데이터셋으로 기존 모델 재학습 스크립트
---
#### 실행 방법 
  1. 환경 설정   
    `pip install -r requirements.txt`

  3. 편향 탐지 프롬프트 생성
      ```bash
      python preprocess/preprocess.py
      python preprocess/get_prompts.py
      ```
  
  4. 편향 식별
      ```bash
      bash bias-detect/get_triplets.sh
      ```
  
  5. Open AI API를 이용한 데이터셋 생성
      - 데이터셋을 직접 생성하기 위해서는 API 키 필요 
      ```bash
      python generate/generate_sentence.py
      ```
      - 이미 생성된 문장은 JSON 파일에서 확인 가능
      ```bash
      ./generate/generated_sentence.json
      ```
  
  6. 모델 재학습
      ```bash
      bash train/run_eps_cutoff.sh
      ```
      
  7. 모델 평가
     - [bias-bench](https://github.com/McGill-NLP/bias-bench)에서 구현된 StereoSet benchmark를 이용
       ```bash
       git clone https://github.com/McGill-NLP/bias-bench.git
       ```
