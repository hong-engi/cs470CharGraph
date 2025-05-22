# --------------------------------------------------
# 0. 공통 의존성 및 유틸 함수(변경 없음)
# --------------------------------------------------
import spacy, time, csv, re, itertools, os
import networkx as nx
from collections import defaultdict
from rapidfuzz import fuzz
from allennlp.predictors.predictor import Predictor
import allennlp_models.coref
import re
from collections import Counter

# ----------------------
# 설정값
# ----------------------
NUM_PARTS = 1              # 텍스트를 몇 등분할지
MAX_CHUNK_WORDS = 1000     # 청크 단어 수 상한
INPUT_FILE = "data/novel/test_novel.txt"
FOLDER_NAME = "test_4"

import os
os.makedirs(FOLDER_NAME, exist_ok=True)

OUT_PREFIX = f"{FOLDER_NAME}/{FOLDER_NAME}_part_"
FINAL_CSV = f"{FOLDER_NAME}/{FOLDER_NAME}.csv"
COREF_MODEL = "coref-model.tar.gz"

def save_graph_edges_to_csv(G, filename):
    with open(filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target", "weight"])
        for u, v, data in G.edges(data=True):
            writer.writerow([u, v, data.get("weight", 1)])


def normalize(name):
    """
    이름을 소문자로 변환, 구두점·소유격 제거, 
    완전 중복된 단어(ex: 'Charley Charley')는 하나로 축소
    """
    # 1) 기본 정규화
    name = name.lower()
    name = re.sub(r"[\"',\.\(\)]", "", name)
    name = re.sub(r"'s$", "", name)
    name = re.sub(r"\s+", " ", name).strip()

    # 2) 중복 토큰 축소
    tokens = name.split()
    if len(tokens) > 1 and len(set(tokens)) == 1:
        # ex: ["charley", "charley"] → "charley"
        name = tokens[0]

    return name
# --- Union–Find -------------------------------------------
class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])   # path compression
        return self.parent[x]

    def union(self, a, b):
        pa, pb = self.find(a), self.find(b)
        if pa != pb:
            self.parent[pb] = pa

    def clusters(self):
        out = defaultdict(set)
        for x in self.parent:
            out[self.find(x)].add(x)
        return out
# ----------------------------------------------------------

def cluster_names(names, threshold=90, name_counts=None):
    """
    ① fuzzy ≥ threshold  OR
    ② 단어 경계 substring  -> 같은 집합으로 Union
    최종적으로 연결된 컴포넌트마다 canonical 이름을 지정한다.
    """
    uf = UnionFind(names)
    name_list = list(names)

    for i, a in enumerate(name_list):
        for b in name_list[i + 1:]:
            if fuzz.token_sort_ratio(a, b) >= threshold or \
               re.search(rf"{re.escape(a)}", b) or \
               re.search(rf"{re.escape(b)}", a):
                uf.union(a, b)

    groups = {}
    for cluster in uf.clusters().values():
        if name_counts:
            # 사용 횟수가 가장 많은 이름
            canonical = max(cluster, key=lambda n: (name_counts.get(n, 0), -len(n)))
        else:
            # fallback: 가장 짧은 다단어 이름 우선
            multi = [n for n in cluster if len(n.split()) > 1]
            canonical = min(multi or cluster, key=len)

        for n in cluster:
            groups[n] = canonical

    return groups



def extract_characters(text):
    doc = nlp(text)
    return {ent.text for ent in doc.ents if ent.label_ == "PERSON"}


def resolve_coreferences(text):
    return predictor.coref_resolved(text)

def build_interaction_matrix_window(text, valid_chars, window_size=3, stride=1, debug = True):
    """
    window_size 문장 단위 슬라이딩 윈도우로 인물 동시등장을 계산합니다.
    두 인물이 같은 윈도우 안에 함께 등장하면 가중치를 +1 합니다.

    Parameters
    ----------
    text : str
    valid_chars : set[str]   # 이미 추출한 인물 이름(원본 표기)
    window_size : int        # m 문장
    stride : int             # 윈도우 이동 간격(기본 1 문장)

    Returns
    -------
    matrix : dict[dict[int]] # 상호작용 매트릭스
    """
    matrix = defaultdict(lambda: defaultdict(int))

    # 추가: 상호작용 문장 로그 저장용 (선택적)
    interaction_log = defaultdict(list)

    # 1) 문장 단위로 자르고, 각 문장마다 등장 인물 집합 저장
    doc = nlp(text)
    sentences = list(doc.sents)
    sent_char_lists = []
    for sent in sentences:
        chars_here = {ent.text for ent in sent.ents
                      if ent.label_ == "PERSON" and ent.text in valid_chars}
        sent_char_lists.append(chars_here)

    # 2) 윈도우 처리
    n = len(sent_char_lists)
    for start in range(0, n - window_size + 1, stride):
        win_chars = set().union(*sent_char_lists[start : start + window_size])
        if len(win_chars) < 2:
            continue  # 인물 2명 이상이 있을 때만 처리

        # 윈도우에 포함된 문장들을 적절히 정리
        window_sents = [str(sentences[start + i]).strip() for i in range(window_size)]
        win_text = "\n".join(window_sents)  # 줄바꿈으로 명확히 나눔``

        for c1, c2 in itertools.combinations(win_chars, 2):
            matrix[c1][c2] += 1
            matrix[c2][c1] += 1

            if debug:
                print("\n[상호작용 발생]")
                print(f"인물 : {c1}, {c2}")
                print(f"문장 : \"{win_text}\"\n")

                # 선택적으로 저장도 가능
                interaction_log[(c1, c2)].append(win_text)
    if debug:
        return matrix, interaction_log
    else:
        return matrix



def split_text_into_chunks(text, max_words):
    paragraphs = text.split("\n\n")
    chunks, curr, curr_len = [], "", 0
    for para in paragraphs:
        words = para.split()
        if curr_len + len(words) > max_words:
            chunks.append(curr.strip())
            curr, curr_len = para, len(words)
        else:
            curr += "\n\n" + para
            curr_len += len(words)
    if curr:
        chunks.append(curr.strip())
    return chunks


def create_cn(matrix):
    G = nx.Graph()
    for c1, nbrs in matrix.items():
        for c2, w in nbrs.items():
            if w > 0:
                G.add_edge(c1, c2, weight=w)
    return G

def create_graph_from_matrix(matrix):
    """
    matrix: {name1: {name2: w, ...}, ...}
    → networkx.Graph
    """
    G = nx.Graph()
    for c1, nbrs in matrix.items():
        for c2, w in nbrs.items():
            if w > 0:
                G.add_edge(c1, c2, weight=w)
    return G

def merge_graph_by_mapping(G_raw, mapping):
    G_new = nx.Graph()
    for u, v, data in G_raw.edges(data=True):
        if u not in mapping or v not in mapping:     # ← 추가
            continue
        cu, cv = mapping[u], mapping[v]
        if cu == cv:
            continue
        w = data.get("weight", 1)
        if G_new.has_edge(cu, cv):
            G_new[cu][cv]["weight"] += w
        else:
            G_new.add_edge(cu, cv, weight=w)
    return G_new

# ----------------------
# 메인 파이프라인
# ----------------------
start = time.time()

small_version = True
if not small_version:
    nlp = spacy.load("en_core_web_trf")
else:
    nlp = spacy.load("en_core_web_sm")

print(f"[0] spaCy 모델 로드: {time.time() - start:.2f}s")

predictor = Predictor.from_path(COREF_MODEL)
print(f"[0] coref 모델 로드: {time.time() - start:.2f}s")

# 1. 텍스트 읽기
st_read = time.time()
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    full_text = f.read()
print(f"[1] 파일 읽기: {time.time() - st_read:.2f}s")

# 2. 텍스트 PARTS 등분
part_len = len(full_text) // NUM_PARTS
text_parts = [full_text[i*part_len : (i+1)*part_len] for i in range(NUM_PARTS-1)]
text_parts.append(full_text[(NUM_PARTS-1)*part_len:])
print(f"[1] 텍스트 {NUM_PARTS} 파트 분할 완료")

# 3. 파트별 처리
final_matrix = defaultdict(lambda: defaultdict(int))

# --------------------------------------------------
# 1. 파트별 처리
# --------------------------------------------------
for idx, part in enumerate(text_parts, 1):
    print(f"\n[파트 {idx}] 시작 --------")
    t_part = time.time()

    # (1) coref → PERSON 엔티티 추출 → 상호작용 매트릭스
    chunks = split_text_into_chunks(part, MAX_CHUNK_WORDS)
    resolved_all, character_set = "", set()

    for j, chunk in enumerate(chunks, 1):
        resolved = resolve_coreferences(chunk)
        resolved_all += resolved + "\n\n"
        character_set.update(extract_characters(resolved))
        print(f"[DEBUG] character_set of {j}: {character_set}")
        print(f" {j}/{len(chunks)} 완료됨 : {time.time() - t_part:.2f}s")

    interaction_debug = True
    result = build_interaction_matrix_window(resolved_all,
                                         character_set,
                                         window_size=3,
                                         stride=1,
                                         debug = interaction_debug)
    if interaction_debug:
        matrix, log = result
        log_path = os.path.join(FOLDER_NAME, f"interaction_log_part{idx}.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            index_count = 1
            for (c1, c2), sentences in log.items():
                f.write("-"*20+"\n")
                f.write(f"{index_count}번째 상호작용\n")
                index_count+=1
                f.write(f"[인물] {c1} - {c2}\n")
                for s in sentences:
                    f.write("-"*20+"\n")
                    f.write(f"문장: {s.strip()}\n")
                f.write("-"*20+"\n")
                f.write("\n\n")
        print(f"  └ 상호작용 로그 저장 완료: {log_path}")
    else:
        matrix = result
    print(f"  ├ 매트릭스 완성 | 인물 {len(character_set)}명 | {time.time() - t_part:.2f}s")

    # (2) 이름 정규화·클러스터링
    raw_names = set(matrix.keys())

    norm = {n: normalize(n) for n in raw_names}
    norm = {k: v for k, v in norm.items() if v}          # "" 제거
    counts = Counter(norm.values())   # 등장 횟수 기반

    groups = cluster_names(set(norm.values()), name_counts=counts)
    mapping = {orig: groups[norm[orig]] for orig in norm if norm[orig] in groups}
    
    # (3) 그래프 생성 → 매핑으로 병합
    G_raw   = create_graph_from_matrix(matrix)
    G_part  = merge_graph_by_mapping(G_raw, mapping)

    # (4) CSV 저장
    part_csv = f"{OUT_PREFIX}{idx}.csv"
    save_graph_edges_to_csv(G_part, part_csv)
    print(f"  └ CSV 저장 완료: {part_csv}")

    # (5) 최종 집계용 매트릭스 업데이트
    for u, v, data in G_part.edges(data=True):
        final_matrix[u][v] += data["weight"]
        final_matrix[v][u] += data["weight"]

# --------------------------------------------------
# 2. 파이널 그래프(전체 파트 합산) 생성
# --------------------------------------------------
raw_final  = set(final_matrix.keys())
norm_f     = {n: normalize(n) for n in raw_final}
counts_f = Counter(norm_f.values())   # 등장 횟수 기반

groups_f = cluster_names(set(norm_f.values()), name_counts=counts_f)

mapping_f  = {orig: groups_f[norm_f[orig]] for orig in raw_final}

G_final_raw = create_graph_from_matrix(final_matrix)
G_final     = merge_graph_by_mapping(G_final_raw, mapping_f)

save_graph_edges_to_csv(G_final, FINAL_CSV)
print(f"[완료] 노드 {G_final.number_of_nodes()} | 엣지 {G_final.number_of_edges()}")
