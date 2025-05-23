#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_character_graphs.py
-------------------------
novel/<genre>/<book_id>.txt  →  books_GNN / books_base CSV
(시간 측정·로그 형식은 원본 스크립트와 동일)
"""

# --------------------------------------------------
# 0. 공통 의존성 및 유틸 함수(원본 그대로)
# --------------------------------------------------
import spacy, time, csv, re, itertools, os, math,json
import torch, gc
import networkx as nx
from collections import defaultdict, Counter
from rapidfuzz import fuzz
from allennlp.predictors.predictor import Predictor
import allennlp_models.coref

print("start")


# GPU 설정
if torch.cuda.is_available():
    spacy.require_gpu()  # spaCy가 GPU 사용하도록 설정
    cuda_device = 0      # AllenNLP용 GPU ID
    print("🟢 GPU detected: spaCy & AllenNLP will use cuda_device=0")
else:
    cuda_device = -1     # CPU fallback
    print("⚪️ GPU not found: using CPU")

NUM_PARTS        = 4
MAX_CHUNK_WORDS  =600
COREF_MODEL      = "coref-model.tar.gz"

# --------------------------------------------------
# 시간 측정: 모델 로드
# --------------------------------------------------
global_start = time.time()
small_version = False
nlp = spacy.load("en_core_web_sm") if small_version else spacy.load("en_core_web_trf")
print(f"[0] spaCy 모델 로드: {time.time() - global_start:.2f}s")

predictor = Predictor.from_path(COREF_MODEL, cuda_device=cuda_device)
print(f"[0] coref 모델 로드: {time.time() - global_start:.2f}s")

# --------------------------------------------------
# CSV 저장 함수
# --------------------------------------------------
def save_graph_edges_to_csv(G, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target", "weight"])

        written = set()
        for u, v, data in G.edges(data=True):
            writer.writerow([u, v, data.get("weight", 1)])
            written.update([u, v])
        for n in G.nodes():
            if n not in written:
                writer.writerow([n, "", 0])

# --------------------------------------------------
# 이름 정규화·클러스터링(원본 그대로)
# --------------------------------------------------
def normalize(name):
    name = name.lower()
    name = re.sub(r"[\"',\.\(\)]", "", name)
    name = re.sub(r"'s$", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    tokens = name.split()
    if len(tokens) > 1 and len(set(tokens)) == 1:
        name = tokens[0]
    return "" if 'chapter' in name else name

class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
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

def cluster_names(names, threshold=90, name_counts=None):
    uf = UnionFind(names)
    name_list = list(names)
    for i, a in enumerate(name_list):
        for b in name_list[i + 1:]:
            if fuzz.token_sort_ratio(a, b) >= threshold or \
               re.search(rf"{re.escape(a)}", b) or re.search(rf"{re.escape(b)}", a):
                uf.union(a, b)
    groups = {}
    for cluster in uf.clusters().values():
        if name_counts:
            canonical = max(cluster, key=lambda n: (name_counts.get(n, 0), -len(n)))
        else:
            multi = [n for n in cluster if len(n.split()) > 1]
            canonical = min(multi or cluster, key=len)
        for n in cluster:
            groups[n] = canonical
    return groups

# --------------------------------------------------
# NLP 기반 함수(원본 그대로)
# --------------------------------------------------
def extract_characters(text):
    doc = nlp(text)
    return {ent.text for ent in doc.ents if ent.label_ == "PERSON"}

def resolve_coreferences(chunk, predictor):
    with torch.no_grad():                # 그래프 저장 방지
        out = predictor.coref_resolved(chunk)
    torch.cuda.empty_cache()             # 캐시 블록 해제
    gc.collect()                         # Python 객체 즉시 수거
    return out

def build_interaction_matrix_window(text, valid_chars, window_size=3, stride=1):
    matrix = defaultdict(lambda: defaultdict(int))
    doc = nlp(text)
    sentences = list(doc.sents)
    sent_char_lists = []
    for sent in sentences:
        chars_here = {ent.text for ent in sent.ents
                      if ent.label_ == "PERSON" and ent.text in valid_chars}
        sent_char_lists.append(chars_here)

    n = len(sent_char_lists)
    for start in range(0, n - window_size + 1, stride):
        win_chars = set().union(*sent_char_lists[start:start + window_size])
        if len(win_chars) < 2:
            continue
        for c1, c2 in itertools.combinations(win_chars, 2):
            matrix[c1][c2] += 1
            matrix[c2][c1] += 1
    return matrix
import re
import re

def split_text_into_chunks(text, max_words=600, max_chars=5000):
    paragraphs = re.split(r'\n\s*\n', text.strip())
    
    chunks = []
    curr_chunk_words = []
    curr_len_chars = 0
    curr_len_words = 0

    for para in paragraphs:
        words = para.strip().split()
        for word in words:
            word_len = len(word) + 1  # +1 for space or newline
            if (curr_len_chars + word_len > max_chars) or (curr_len_words + 1 > max_words):
                # 강제 청크 분리
                if curr_chunk_words:
                    chunks.append(' '.join(curr_chunk_words).strip())
                curr_chunk_words = [word]
                curr_len_chars = len(word) + 1
                curr_len_words = 1
            else:
                curr_chunk_words.append(word)
                curr_len_chars += word_len
                curr_len_words += 1

        # 단락 경계에는 두 줄 바꿈을 넣어 가독성 유지
        if curr_chunk_words and curr_len_chars + 2 <= max_chars:
            curr_chunk_words.append('\n\n')
            curr_len_chars += 2

    # 마지막 청크 추가
    final_text = ' '.join(curr_chunk_words).strip()
    if final_text:
        chunks.append(final_text)

    return chunks

def create_graph_from_matrix(matrix):
    G = nx.Graph()
    for c1, nbrs in matrix.items():
        for c2, w in nbrs.items():
            if w > 0:
                G.add_edge(c1, c2, weight=w)
    return G

def merge_graph_by_mapping(G_raw, mapping):
    G_new = nx.Graph()
    for u, v, data in G_raw.edges(data=True):
        if u not in mapping or v not in mapping:
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

def align_graph_nodes(G_src, node_order):
    G_dst = nx.Graph()
    G_dst.add_nodes_from(node_order)
    for u, v, data in G_src.edges(data=True):
        if u in node_order and v in node_order:
            G_dst.add_edge(u, v, weight=data.get("weight", 1))
    return G_dst

# --------------------------------------------------
# 1. 단일 도서 처리
# --------------------------------------------------
def process_single_book(input_file):
    book_id     = os.path.splitext(os.path.basename(input_file))[0]
    gnn_folder  = os.path.join("books_GNN", book_id)
    out_prefix  = os.path.join(gnn_folder, f"{book_id}_part_")
    final_csv   = os.path.join("books_base", f"{book_id}.csv")

    os.makedirs(gnn_folder, exist_ok=True)
    os.makedirs("books_base", exist_ok=True)

    # 파일 읽기
    st_read = time.time()
    with open(input_file, "r", encoding="utf-8") as f:
        full_text = f.read()
    print(f"[1] 파일 읽기 완료: {time.time() - st_read:.2f}s")

    part_len   = len(full_text) // NUM_PARTS
    text_parts = [full_text[i*part_len:(i+1)*part_len] for i in range(NUM_PARTS-1)]
    text_parts.append(full_text[(NUM_PARTS-1)*part_len:])
    print(f"[1] 텍스트 {NUM_PARTS} 파트 분할 완료")

    final_matrix    = defaultdict(lambda: defaultdict(int))
    part_graphs_raw = []

    for idx, part in enumerate(text_parts, 1):
        print(f"\n[파트 {idx}] 시작 --------")
        t_part = time.time()

        chunks = split_text_into_chunks(part, MAX_CHUNK_WORDS)
        resolved_all, character_set = "", set()
        for j, chunk in enumerate(chunks, 1):
            resolved = resolve_coreferences(chunk, predictor)
            resolved_all += resolved + "\n\n"
            character_set.update(extract_characters(resolved))
            print(f"  {j}/{len(chunks)} 청크 완료 ({time.time() - t_part:.2f}s)")
        matrix = build_interaction_matrix_window(resolved_all, character_set)
        print(f"  ├ 매트릭스 완성 | 인물 {len(character_set)}명 | {time.time() - t_part:.2f}s")

        norm      = {n: normalize(n) for n in matrix if normalize(n)}
        counts    = Counter(norm.values())
        groups    = cluster_names(set(norm.values()), name_counts=counts)
        mapping   = {orig: groups[norm[orig]] for orig in norm}

        G_part = merge_graph_by_mapping(create_graph_from_matrix(matrix), mapping)
        part_graphs_raw.append((idx, G_part))

        for u, v, data in G_part.edges(data=True):
            final_matrix[u][v] += data["weight"]
            final_matrix[v][u] += data["weight"]

    # 전체 그래프
    norm_f   = {n: normalize(n) for n in final_matrix}
    counts_f = Counter(norm_f.values())
    groups_f = cluster_names(set(norm_f.values()), name_counts=counts_f)
    mapping_f = {orig: groups_f[norm_f[orig]] for orig in norm_f}

    G_final_raw = create_graph_from_matrix(final_matrix)
    G_final     = merge_graph_by_mapping(G_final_raw, mapping_f)
    node_order  = list(G_final.nodes())

    # CSV 저장
    for idx, G_raw in part_graphs_raw:
        G_aligned = align_graph_nodes(G_raw, node_order)
        save_graph_edges_to_csv(G_aligned, f"{out_prefix}{idx}.csv")
        print(f"  └ 파트 {idx} CSV 저장 완료")
    save_graph_edges_to_csv(G_final, final_csv)
    print(f"[완료] {book_id} | 노드 {G_final.number_of_nodes()} | 엣지 {G_final.number_of_edges()}")

# --------------------------------------------------
# 2. 전체 폴더 순회
# --------------------------------------------------
PROCESSED_IDS_FILE = "processed_ids.json"

def load_processed_ids():
    if os.path.exists(PROCESSED_IDS_FILE):
        with open(PROCESSED_IDS_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()

def save_processed_ids(processed_ids):
    with open(PROCESSED_IDS_FILE, "w", encoding="utf-8") as f:
        json.dump(list(processed_ids), f, indent=2)

def batch_process_all(root_dir, genre):
    genre_path = os.path.join(root_dir, genre)
    if not os.path.isdir(genre_path):
        print(f"[ERROR] 폴더 없음: {genre_path}")
        return

    processed_ids = load_processed_ids()

    for fname in os.listdir(genre_path):
        if fname.endswith(".txt"):
            if fname in processed_ids:
                print(f"[SKIP] 이미 처리됨: {fname}")
                continue

            input_file = os.path.join(genre_path, fname)
            print(f"\n=== {genre}/{fname} 처리 시작 ===")
            process_single_book(input_file)

            processed_ids.add(fname)
            save_processed_ids(processed_ids)

# --------------------------------------------------
# 3. 실행
# --------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch process character graphs for a given genre.")
    parser.add_argument("--genre", required=True, help="Genre folder name (e.g., 'humorous_stories')")
    parser.add_argument("--root_dir", default="novel", help="Root directory (default: 'novel')")

    args = parser.parse_args()
    batch_process_all(root_dir=args.root_dir, genre=args.genre)
