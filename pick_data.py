import json
import random
import os

# 설정: 경로 및 출력 파일명
PARSED_DIR = os.path.join('data', 'genre', 'parsed')
INPUT_FILE = os.path.join(PARSED_DIR, 'single_genre_books.json')
OUTPUT_FILE = os.path.join(PARSED_DIR, 'smallest5_100.json')

# 1) single_genre_books 불러오기
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    obj = json.load(f)
single_genre_books = obj['single_genre_books']

# 2) 장르별 책 개수 계산 및 오름차순 정렬
genre_counts = {genre: len(ids) for genre, ids in single_genre_books.items()}
sorted_genres = sorted(genre_counts, key=lambda g: genre_counts[g])

# 3) 책 개수가 가장 적은 다섯 장르 선택
smallest5 = sorted_genres[:5]

# 4) 각 장르에서 최대 100권 무작위 샘플링
sample_by_genre = {}
for genre in smallest5:
    ids = single_genre_books[genre]
    if len(ids) >= 100:
        sample = random.sample(ids, 100)
    else:
        sample = ids.copy()
    sample_by_genre[genre] = sample

# 5) JSON으로 저장
output = {
    'smallest5_genres': smallest5,
    'sample_by_genre': sample_by_genre
}

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"Saved smallest5 samples to {OUTPUT_FILE}")
