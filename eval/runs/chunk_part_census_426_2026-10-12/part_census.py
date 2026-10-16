"""Сколько чанков 426-ФЗ не несут номер части статьи (снимок build-n_a_5_83, без LLM)."""

import collections
import re
import sys

import chromadb

DB = sys.argv[1]
SRC = "426-ФЗ.pdf"

ART = re.compile(r"Статья\s+(\d+(?:\.\d+)?)")
PART = re.compile(r"^(\d+(?:\.\d+)?)\.\s+(?=[А-ЯЁ])", re.M)  # «2. Работник обязан…»
FAKE = re.compile(r"^(\d+)\.\s+[а-я]\)", re.M)  # «9. а)» — номер списка от Docling

col = chromadb.PersistentClient(path=DB).get_collection("documents")
r = col.get(where={"source": SRC}, include=["metadatas", "documents"])
items = sorted(zip(r["metadatas"], r["documents"]), key=lambda x: x[0]["chunk_id"])

by_art = collections.defaultdict(list)
no_art = 0
for m, d in items:
    a = ART.search(m.get("heading_path", ""))
    if not a:
        no_art += 1
        continue
    body = d.split("\n", 1)[1] if "\n" in d else ""
    by_art[a.group(1)].append((m["chunk_id"], body))

meta_part = sum(
    1 for m, _ in items if any("част" in k.lower() or k == "part" for k in m)
)

cnt = collections.Counter()
examples = collections.defaultdict(list)
for art, chunks in by_art.items():
    has_parts = any(PART.search(b) for _, b in chunks)
    last = None
    for cid, body in chunks:
        if not has_parts:
            cnt["статья без частей"] += 1
            continue
        starts = PART.match(body)
        inside = PART.search(body)
        if starts:
            k = "часть видна в начале"
        elif inside:
            k = "часть видна в середине (начало чанка — хвост предыдущей)"
        elif last:
            k = "части нет в тексте, восстановима по порядку"
        else:
            k = "части нет и не восстановима"
        if FAKE.search(body):
            k += " + ложный номер «N. а)»"
        cnt[k] += 1
        if len(examples[k]) < 2:
            examples[k].append(f"ст.{art} chunk {cid}: {body[:90]!r}")
        found = PART.findall(body)
        if found:
            last = found[-1]

print(f"Чанков 426-ФЗ: {len(items)}; без «Статья N» в heading_path: {no_art}")
print(f"Статей: {len(by_art)}; ключей метаданных с частью: {meta_part}")
for k, v in cnt.most_common():
    print(f"{v:4d}  {k}")
    for e in examples[k]:
        print("       ", e)
