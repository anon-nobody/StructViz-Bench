# StructViz-Bench — 인간 검증 (Task A: 정답 판독성 감사)

## 목적
각 문항의 **정답(ground_truth)이, 주어진 이미지만 보고** 명확하고 모호하지 않게 맞는지 판정합니다.
(정답은 원본 데이터로부터 프로그램적으로 계산되어 데이터 기준으로는 이미 정확합니다. 여기서는
**렌더된 이미지 수준에서 답을 읽어낼 수 있는지**를 별도로 검증합니다.)

## 권장 방법 — `annotate.html` (약 1시간)
1. 압축을 푼 폴더에서 **`annotate.html`을 더블클릭**해 브라우저(Chrome/Edge/Safari)로 엽니다. 설치·인터넷 불필요.
2. 상단에서 **본인 평가자 번호(1/2/3)** 를 선택합니다.
3. 한 화면에 이미지·질문·제시된 정답이 나옵니다. 질문은 **영어 원문이 기준**이며(모델에게 제시된 그대로), 바로 아래 회색 줄에 **참고용 한국어 번역**이 함께 표시됩니다. **Correct / Ambiguous / Incorrect** 버튼(또는 키 `1` `2` `3`)으로 판정하세요.
   - Correct를 고르면 자동으로 다음 문항으로 넘어갑니다. Ambiguous/Incorrect는 `notes`에 간단한 이유를 적어 주세요.
   - (선택) `your_answer`에 이미지만 보고 낸 본인 답을 적으면 인간 정확도 상한도 계산됩니다.
   - 이미지를 클릭하면 원본 크기로 새 탭에서 열립니다. `←` `→`로 이동, 하단 번호판으로 점프.
4. 진행 상황은 **같은 브라우저에 자동 저장**되므로 중간에 닫고 나중에 이어서 해도 됩니다.
5. 100문항이 끝나면 **"CSV 내보내기"** 를 눌러 다운로드된 `ratings_annotatorN.csv`를 회신합니다.

## 대안 — 스프레드시트로 직접 입력
1. 배정받은 시트 `ratings_annotatorN.csv`를 엽니다 (N = 1, 2, 3 중 본인 번호).
2. 각 행마다:
   - `image` 경로의 그림을 엽니다 (예: `images/H000_tabular_table_image.png`).
   - `question`(질문)과 `ground_truth`(제시된 정답)를 봅니다.
   - **`rating` 칸**에 아래 셋 중 하나를 정확히 입력합니다:
     - `Correct` — 이미지만 보고 ground_truth가 **명확히 맞다**.
     - `Ambiguous` — 답이 방어 가능하지만 이미지/질문이 **다른 해석도 허용**한다.
     - `Incorrect` — ground_truth가 **틀려 보인다**.
   - (선택) `your_answer` — 이미지만 보고 **본인이 낸 답**. 채우면 인간 정확도 상한도 계산됩니다.
   - `notes` — Ambiguous/Incorrect를 고른 경우 간단한 이유.
3. **다른 평가자와 상의하지 마세요.** 독립적으로 판정해야 합니다 (합의율·Fleiss κ 산출).
4. 다 채운 CSV를 반환합니다. 소요 시간 ≈ 1.5–2시간(100문항).

## 구성
- `annotate.html` — **판정 도구** (더블클릭으로 실행, 오프라인, 자동 저장, CSV 내보내기).
- `items.jsonl` — 100문항 메타데이터(참고용).
- `images/` — 100개 렌더 이미지 (자체 동봉, 별도 접근 불필요).
- `ratings_annotator1.csv` / `2` / `3` — 평가자별 시트 (동일 100문항).
- `assemble_ratings.py`, `human_eval_aggregate.py` — 집계용 (아래).
- 표본: 모달 균형 34 tabular / 33 timeseries / 33 graph, 포맷·태스크 분산.

## 집계 (3명 회수 후)
```bash
# 1) 3개 시트를 long 형식으로 병합
python assemble_ratings.py --sheets ratings_annotator1.csv ratings_annotator2.csv ratings_annotator3.csv --out ratings_long.csv
# 2) 합의율(%correct/ambiguous/incorrect) + Fleiss' kappa + 논문용 문장 출력
python human_eval_aggregate.py --ratings ratings_long.csv
```
출력된 "REBUTTAL SENTENCE"를 논문 Appendix B에 그대로 넣으면 됩니다.
목표(프로토콜): majority-vote 기준 **≥95% Correct**면 벤치마크 정답 품질이 검증됩니다.
