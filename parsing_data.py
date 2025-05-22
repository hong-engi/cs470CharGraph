import json
import glob
import os

# 입력·출력 경로 설정
INPUT_DIR  = 'data/genre'
OUTPUT_DIR = os.path.join(INPUT_DIR, 'parsed')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# genre 파일 목록(정렬) 가져오기
paths = sorted(glob.glob(os.path.join(INPUT_DIR, 'genre-*.json')))

genre_idx     = {}
genre_to_ids  = {}

# 1) 각 장르에 인덱스 부여 및 True인 ID 추출
for idx, path in enumerate(paths):
    # 파일명에서 genre 이름만 추출
    name  = os.path.basename(path)                   # ex. 'genre-19th_century.json'
    genre = name[len('genre-'):-len('.json')]        # ex. '19th_century'
    
    genre_idx[genre] = idx
    
    # JSON 로드 후 True인 A 부분만 int로 추출
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)['data']
    ids = [int(code.split('_',1)[0]) for code, flag in data if flag]
    
    genre_to_ids[genre] = ids

# 2) 각 ID가 속한 장르 인덱스 집합 생성
books = {}
for genre, ids in genre_to_ids.items():
    idx = genre_idx[genre]
    for bid in ids:
        books.setdefault(bid, set()).add(idx)

# JSON으로 직렬화하기 위해 set → list 변환
books_json = {str(bid): sorted(list(idxs)) for bid, idxs in books.items()}

# 최종 구조
output = {
    'genre_idx': genre_idx,
    'books':     books_json
}

# 파일로 저장
with open(os.path.join(OUTPUT_DIR, 'genre_books.json'), 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"Saved: {os.path.join(OUTPUT_DIR, 'genre_books.json')}")

single_genre_books = {genre: [] for genre in genre_idx}  # {장르명: [book_id,...]}
for book_id, idxs in books.items():
    if len(idxs) == 1:
        # 유일한 인덱스를 장르명으로 되돌리기
        genre_name = next(g for g, i in genre_idx.items() if i == next(iter(idxs)))
        single_genre_books[genre_name].append(book_id)

# 4) 장르별 개수 세기
single_counts = {genre: len(id_list) for genre, id_list in single_genre_books.items()}

# — 결과 출력 예시
print("한 장르에만 속한 책들:")
for g, lst in single_genre_books.items():
    print(f"  {g} {single_counts[g]}권")

# — JSON으로 저장하고 싶다면
OUTPUT_DIR = os.path.join('data/genre', 'parsed')
with open(os.path.join(OUTPUT_DIR, 'single_genre_books.json'), 'w', encoding='utf-8') as f:
    json.dump({
        'single_genre_books': single_genre_books,
        'single_counts': single_counts
    }, f, ensure_ascii=False, indent=2)
