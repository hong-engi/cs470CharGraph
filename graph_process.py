#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
canonical_aggregate.py
----------------------
books_GNN/<id>/<id>_part_*.csv  ➜  동일 canonical 이름 체계로
  1) 모든 파트 CSV 읽어 전역 canonical 매핑 산출
  2) 그 매핑으로 파이널 그래프 구축 → books_base/<id>.csv 저장
  3) 같은 매핑을 각 파트 CSV에 역전파(노드 이름 교정·가중치 병합)
또한 엣지 수가 EDGE_THRESHOLD 미만인 파트는 <id>_report.txt 로 경고 기록
"""

import os, csv, re, json
from collections import defaultdict, Counter
from rapidfuzz import fuzz

BOOKS_GNN   = "books_GNN"
BOOKS_BASE  = "books_base"
EDGE_THRESHOLD = 3
HEADER = ["source", "target", "weight"]

# ------------------------------------------------------------------
# 이름 정규화 & 클러스터링
# ------------------------------------------------------------------

def normalize(name: str) -> str:
    name = name.lower()
    name = re.sub(r"[\"',\.\(\)]", "", name)
    name = re.sub(r"'s$", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    toks = name.split()
    if len(toks) > 1 and len(set(toks)) == 1:
        name = toks[0]
    return name

class UnionFind:
    def __init__(self, items):
        self.p = {x: x for x in items}
    def find(self, x):
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]
    def union(self, a, b):
        pa, pb = self.find(a), self.find(b)
        if pa != pb:
            self.p[pb] = pa
    def clusters(self):
        out = defaultdict(set)
        for x in self.p:
            out[self.find(x)].add(x)
        return out

def cluster_names(names, threshold=90, name_counts=None):
    uf = UnionFind(names)
    nlist = list(names)
    for i, a in enumerate(nlist):
        for b in nlist[i+1:]:
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

# ------------------------------------------------------------------
# 핵심 로직
# ------------------------------------------------------------------
cnt=0
def aggregate_one(book_id: str):
    global cnt
    part_dir = os.path.join(BOOKS_GNN, book_id)
    if not os.path.isdir(part_dir):
        print(f"[WARN] 폴더 없음: {part_dir}")
        return

    # 1) 파트 읽기 → raw edge·node 수집
    all_edges = []  # (src, tgt, w)
    name_freq = Counter()
    part_files = sorted([f for f in os.listdir(part_dir) if f.startswith(f"{book_id}_part_") and f.endswith('.csv')])
    if not part_files:
        print(f"[WARN] 파트 CSV 없음: {part_dir}")
        return

    for f in part_files:
        with open(os.path.join(part_dir, f), newline='', encoding='utf-8') as fh:
            for row in csv.DictReader(fh):
                s, t, w = row['source'].strip(), row['target'].strip(), int(row['weight'])
                if not s:
                    continue
                if t:
                    all_edges.append((s, t, w))
                    name_freq[s] += 1; name_freq[t] += 1
                else:
                    name_freq[s] += 1  # 고립 노드도 빈도 집계

    if not all_edges:
        print(f"[WARN] 엣지 없음: {book_id}")
        return

    # 2) 글로벌 canonical 매핑 생성
    norm_names = {n: normalize(n) for n in name_freq}
    groups = cluster_names(set(norm_names.values()), name_counts=name_freq)
    mapping = {orig: groups[norm_names[orig]] for orig in norm_names}

    # 헬퍼
    def remap_pair(u, v):
        cu, cv = mapping.get(u, u), mapping.get(v, v)
        return (cu, cv) if cu <= cv else (cv, cu)

    # 3) 파이널 그래프 구축
    final_edges = defaultdict(int)
    nodes = set()
    for u, v, w in all_edges:
        cu, cv = remap_pair(u, v)
        if cu == cv:
            continue  # self-loop 제거
        final_edges[(cu, cv)] += w
        nodes.update([cu, cv])

    os.makedirs(BOOKS_BASE, exist_ok=True)
    final_csv = os.path.join(BOOKS_BASE, f"{book_id}.csv")
    with open(final_csv, 'w', newline='', encoding='utf-8') as fh:
        wr = csv.writer(fh); wr.writerow(HEADER)
        written = set()
        for (u, v), w in final_edges.items():
            wr.writerow([u, v, w]); written.update([u, v])
        for n in nodes:
            if n not in written:
                wr.writerow([n, '', 0])
    print(f"✅  파이널 저장: {final_csv}  (edges={len(final_edges)})")


    # 4) 각 파트 CSV 재작성 + 경고 보고
    warn_lines = []
    edge_sum_check = 0
    for f in part_files:
        path = os.path.join(part_dir, f)
        part_edge = defaultdict(int)
        part_nodes = set()
        with open(path, newline='', encoding='utf-8') as fh:
            for row in csv.DictReader(fh):
                s, t, w = row['source'].strip(), row['target'].strip(), int(row['weight'])
                if not s:
                    continue
                if not t:
                    part_nodes.add(mapping.get(s, s))
                    continue
                cu, cv = remap_pair(s, t)
                part_edge[(cu, cv)] += w
                part_nodes.update([cu, cv])
                edge_sum_check += w

        # 엣지 수 경고
        if len(part_edge) < EDGE_THRESHOLD:
            warn_lines.append(f"⚠️  {f}: edge count = {len(part_edge)}")

        with open(path, 'w', newline='', encoding='utf-8') as fh:
            wr = csv.writer(fh); wr.writerow(HEADER)
            written = set()
            for (u, v), w in part_edge.items():
                wr.writerow([u, v, w]); written.update([u, v])
            for n in part_nodes:
                if n not in written:
                    wr.writerow([n, '', 0])
        
    final_sum = sum(final_edges.values())
    assert final_sum == edge_sum_check, f"엣지 총합 불일치: final={final_sum}, parts={edge_sum_check}"

    if warn_lines:
        rpt = os.path.join(BOOKS_BASE, f"{book_id}_report.txt")
        with open(rpt, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(warn_lines))
        print(f"⚠️  경고 저장: {rpt}")
        cnt+=1


def aggregate_all():
    for sub in sorted(os.listdir(BOOKS_GNN)):
        if os.path.isdir(os.path.join(BOOKS_GNN, sub)):
            aggregate_one(sub)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--id', help='특정 book_id만 처리 (미지정 시 전체)')
    args = ap.parse_args()

    if args.id:
        aggregate_one(args.id)
    else:
        aggregate_all()
    print(cnt)