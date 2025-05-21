# --------------------------------------------------
# 0. 공통 의존성 및 유틸 함수(변경 없음)
# --------------------------------------------------
import spacy, time, csv, re, itertools, os
import networkx as nx
from collections import defaultdict
from rapidfuzz import fuzz
from allennlp.predictors.predictor import Predictor
import allennlp_models.coref


# ----------------------
# 설정값
# ----------------------
NUM_PARTS = 4              # 텍스트를 몇 등분할지
MAX_CHUNK_WORDS = 1000     # 청크 단어 수 상한
INPUT_FILE = "data/novel/11051.txt"
FOLDER_NAME = "fuzzy_substring_2"

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

from rapidfuzz import process, fuzz
import re

def cluster_names(names, threshold=90):
    """
    names: normalize() 처리된 이름들의 iterable
    1) fuzzy 매칭(score ≥ threshold)이면 병합
    2) 토큰 단위 substring(단어 경계) 포함 관계이면 병합
    대표(canonical)는 가장 긴 이름으로 선택
    """
    groups = {}
    name_list = list(names)
    for nm in name_list:
        if nm in groups:
            continue

        cluster = {nm}
        for other in name_list:
            if other == nm:
                continue

            # 1) fuzzy 매칭
            score = fuzz.token_sort_ratio(nm, other)
            if score >= threshold:
                cluster.add(other)
                continue

            # 2) 토큰 단위 substring 매칭
            # 예: 'charley' in 'charley fred' 또는 반대
            if re.search(rf"\b{re.escape(nm)}\b", other) or \
               re.search(rf"\b{re.escape(other)}\b", nm):
                cluster.add(other)

        # 대표 이름은 가장 짧은(normalized) 문자열
        canonical = min(cluster, key=len)
        for m in cluster:
            groups[m] = canonical

    return groups



def extract_characters(text):
    doc = nlp(text)
    return {ent.text for ent in doc.ents if ent.label_ == "PERSON"}


def resolve_coreferences(text):
    return predictor.coref_resolved(text)


def build_interaction_matrix(text, valid_chars):
    matrix = defaultdict(lambda: defaultdict(int))
    doc = nlp(text)
    for sent in doc.sents:
        sent_chars = [ent.text for ent in sent.ents if ent.label_ == "PERSON" and ent.text in valid_chars]
        for c1, c2 in itertools.combinations(set(sent_chars), 2):
            matrix[c1][c2] += 1
            matrix[c2][c1] += 1
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
    """
    G_raw  : 클러스터링 전 원본 그래프
    mapping: {원본이름 → canonical이름}
    → canonical 이름 기준으로 병합된 그래프
    """
    G_new = nx.Graph()
    for u, v, data in G_raw.edges(data=True):
        cu, cv = mapping[u], mapping[v]
        if cu == cv:          # 같은 클러스터 내부 연결은 무시
            continue
        w = data["weight"]
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

    matrix = build_interaction_matrix(resolved_all, character_set)
    print(f"  ├ 매트릭스 완성 | 인물 {len(character_set)}명 | {time.time() - t_part:.2f}s")

    # (2) 이름 정규화·클러스터링
    raw_names = set(matrix.keys())
    norm      = {n: normalize(n) for n in raw_names}
    groups    = cluster_names(set(norm.values()))
    mapping   = {orig: groups[norm[orig]] for orig in raw_names}  # 원본→대표

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
groups_f   = cluster_names(set(norm_f.values()))
mapping_f  = {orig: groups_f[norm_f[orig]] for orig in raw_final}

G_final_raw = create_graph_from_matrix(final_matrix)
G_final     = merge_graph_by_mapping(G_final_raw, mapping_f)

save_graph_edges_to_csv(G_final, FINAL_CSV)
print(f"[완료] 노드 {G_final.number_of_nodes()} | 엣지 {G_final.number_of_edges()}")
