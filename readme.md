# Character Network Graph Generator

이 프로젝트는 구텐베르그(Gutenberg) 소설 데이터를 기반으로, 인물 관계 그래프를 생성하고 분석하는 도구입니다. 소설 텍스트를 기반으로 전체 및 부분별 인물 그래프를 만들고, 이를 분류 또는 시각화할 수 있습니다.

## 디렉토리 구조

```
.
├── books_base/              # 각 소설의 전체 인물 그래프 (CSV)
├── books_GNN/               # 각 소설을 4개의 파트로 나눈 인물 그래프 (CSV)
├── data/
│   └── genre/
│       ├── genre-data/      # 구텐베르그 장르 데이터를 JSON으로 저장
│       └── parsed/          # 가공된 장르 데이터 및 샘플링 결과 (e.g., smallest5_*.json)
├── novel/                   # 서론 및 후기를 제거한 정제된 소설 텍스트
```

## 주요 스크립트

| 파일명                  | 설명                                                      |
| -------------------- | ------------------------------------------------------- |
| `graph_process.py`   | `books_GNN`의 파트 그래프들을 통합하여 `books_base`의 전체 그래프를 생성합니다. |
| `graph_visualize.py` | 생성된 그래프를 시각화하는 도구입니다.                                   |
| `networkGen*.py`      | 전체 파이프라인을 실행하여 소설을 그래프로 변환하는 메인 코드입니다.                 |
| `novel_get.py`       | 구텐베르그에서 소설 원문을 다운로드합니다.                                 |
| `pick_data*.py`       | 특정 장르에서 무작위로 소설을 선택합니다.                                 |
| `slice_novels.py`    | 소설에서 서론과 끝부분(면책 조항 등)을 제거합니다. 자동 처리 후 일부 수동 정제 필요합니다.   |

## 데이터 처리 흐름

1. **소설 다운로드**: `novel_get.py`를 사용하여 원문을 수집합니다.
2. **텍스트 정제**: `slice_novels.py`를 통해 서론 및 끝말 제거 (불완전할 경우 수동 편집).
3. **데이터 샘플링**: `pick_data.py`를 통해 특정 장르에서 무작위로 소설을 선정 (`smallest5_*.json`).
4. **그래프 생성**: `networkGen.py`로 인물 추출 및 관계 분석 후 그래프 생성 (`books_GNN`, `books_base`).
5. **그래프 통합**: `graph_process.py`를 통해 전체 그래프 구성.
6. **시각화 (선택)**: `graph_visualize.py`로 그래프를 시각적으로 확인 가능.

## 요구 사항

* Python 3.9.22
* 주요 라이브러리: `networkx`, `spacy`, `allennlp`, `rapidfuzz` 등
* GPU 사용을 권장합니다 (spaCy와 AllenNLP의 성능 개선)
