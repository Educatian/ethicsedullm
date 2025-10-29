# 🚀 Google Colab 실행 가이드

이 가이드는 Google Colab에서 AI Ethics LLM을 훈련하는 방법을 설명합니다.

## 📋 사전 준비

### 1. Hugging Face 계정 생성 및 Llama 3.1 액세스
1. https://huggingface.co/ 가입
2. https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct 방문
3. **"Agree and access repository"** 클릭 (라이센스 동의)
4. https://huggingface.co/settings/tokens 에서 토큰 생성
   - Token type: **Write** 선택
   - 토큰 복사해두기

### 2. Google Colab 계정
- Gmail 계정만 있으면 됨 (무료)
- Pro/Pro+는 선택사항 (A100 GPU 사용 가능)

---

## 🎯 실행 방법

### Option A: Colab에서 직접 열기 (추천)

1. 아래 링크 클릭:

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Educatian/ethicsedullm/blob/main/AI_Ethics_LLM_Training_Colab.ipynb)

2. **Runtime → Change runtime type → GPU**
   - Free: T4 (15GB VRAM) - 충분함
   - Pro: V100 (16GB) 또는 A100 (40GB) - 더 빠름

3. 셀을 위에서부터 순서대로 실행 (Shift + Enter)

### Option B: 노트북 파일 업로드

1. GitHub에서 `AI_Ethics_LLM_Training_Colab.ipynb` 다운로드
2. https://colab.research.google.com/ 접속
3. **File → Upload notebook**
4. 다운로드한 `.ipynb` 파일 선택
5. **Runtime → Change runtime type → GPU**

---

## ⏱️ 예상 소요 시간

| GPU 타입 | 훈련 시간 | 비용 |
|---------|---------|------|
| T4 (Free) | ~2-3시간 | 무료 |
| V100 (Pro) | ~1.5시간 | $10/월 |
| A100 (Pro+) | ~1시간 | $50/월 |

**무료 T4로도 충분히 가능합니다!**

---

## 📝 실행 순서

노트북의 각 섹션을 순서대로 실행:

### 1️⃣ Environment Setup
- GPU 확인
- 패키지 설치 (~5분)

### 2️⃣ Clone Repository
- GitHub에서 프로젝트 클론

### 3️⃣ Hugging Face Login
- 준비한 HF 토큰 입력
- Llama 3.1 다운로드 권한 획득

### 4️⃣ Prepare Data
- 30개 AI 윤리 데이터셋 생성 (~1분)

### 5️⃣ Training (중요!)
- QLoRA 파인튜닝 시작
- ⚠️ **탭을 닫지 마세요!**
- T4: 2-3시간, A100: 1시간

### 6️⃣ Evaluation
- 모델 성능 평가 (~5분)

### 7️⃣ Demo
- Gradio 인터페이스로 테스트
- 질문하고 답변 확인!

### 8️⃣ Save (선택)
- Hugging Face에 업로드
- 또는 로컬에 다운로드

---

## 💡 팁 & 주의사항

### ✅ DO
- **GPU 런타임 선택** 필수
- 훈련 중 탭 열어두기
- 중간중간 결과 확인
- 모델 저장/다운로드

### ❌ DON'T
- 훈련 중 탭 닫기 (진행상황 손실)
- CPU 런타임 사용 (너무 느림)
- 무료 GPU 시간 제한 초과 (12시간)

### 💰 무료 Colab 제한
- **연속 실행**: 최대 12시간
- **유휴 시간**: 90분 후 연결 해제
- **일일 제한**: ~12-24시간 GPU 사용 가능

**해결책**:
- Pro 구독 ($10/월) → 24시간 연속, 더 좋은 GPU
- 또는 여러 세션으로 나눠서 훈련

---

## 🔧 문제 해결

### "No GPU available" 에러
→ **Runtime → Change runtime type → GPU** 선택

### "Out of memory" 에러
→ 노트북에서 `per_device_train_batch_size=2`로 줄이기

### Llama 3.1 다운로드 안됨
→ HF에서 라이센스 동의했는지 확인
→ 토큰 권한이 'Write'인지 확인

### 훈련 중 연결 끊김
→ 탭을 활성 상태로 유지
→ 또는 Pro 구독

### 무료 GPU 시간 소진
→ 다음 날 다시 시도
→ 또는 Pro 구독

---

## 📊 기대 결과

훈련 완료 후:

```
Average Combined Score:  0.75-0.85
Average Keyword Score:   0.70-0.80
Average Rubric Score:    0.75-0.85

Rubric Breakdown:
  accuracy          : 0.80
  completeness      : 0.75
  clarity           : 0.80
  ethical_awareness : 0.70
  actionability     : 0.65
```

---

## 📤 다음 단계

훈련 완료 후:

1. **모델 다운로드**
   ```python
   !zip -r model.zip ai_ethics_llm_final/
   # Colab 왼쪽 폴더에서 다운로드
   ```

2. **Hugging Face 업로드**
   ```python
   # 노트북의 8️⃣ 섹션 참고
   ```

3. **로컬에서 사용**
   ```python
   from transformers import AutoModelForCausalLM
   from peft import PeftModel

   base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
   model = PeftModel.from_pretrained(base, "path/to/ai_ethics_llm_final")
   ```

---

## 🆘 도움이 필요하면

- GitHub Issues: https://github.com/Educatian/ethicsedullm/issues
- Colab FAQ: https://research.google.com/colaboratory/faq.html
- Hugging Face Docs: https://huggingface.co/docs

---

**Happy Training! 🚀**
