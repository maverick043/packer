#!/usr/bin/env python3
"""
End-to-end check of the RAG backend exactly as Open WebUI calls it (through LiteLLM).
Standard library only, so it runs inside the Open WebUI pod (python3 is there):

  kubectl exec -i deploy/open-webui -- python3 - \
      --base http://litellm.litellm.svc.cluster.local:4000/v1 --key "$LITELLM_KEY" \
      < verify_rag_stack.py

Checks
  1. models are registered in LiteLLM
  2. chat (non-thinking) answers without reasoning
  3. a ~16k-token prompt is accepted (typical RAG request; catches low max-model-len)
  4. thinking alias still thinks (optional)
  5. embeddings: 4096-dim, and similarity scores match NVIDIA's model card when the
     "query: "/"passage: " prefixes are used -> catches wrong pooling / missing prefixes
  6. reranker returns the relevant passage first (skip with --rerank-model none)
Exit code 0 = all passed.
"""
import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request

# From the Nemotron-3-Embed-8B-BF16 model card (vLLM example).
QUERIES = [
    "Write a Python function that counts the frequency of each element in a list of lists.",
    "Write a function that orders a dictionary with tuple keys by the product of each key's tuple values.",
    "What symptoms and common triggers help distinguish eczema from other inflammatory skin conditions?",
    "How can someone reduce exposure to pollen during allergy season?",
]
DOCUMENTS = [
    "def frequency_lists(list1):\n    flattened = [item for sublist in list1 for item in sublist]\n    counts = {}\n    for item in flattened:\n        if item in counts:\n            counts[item] += 1\n        else:\n            counts[item] = 1\n    return counts",
    "def sort_dict_item(test_dict):\n    return {key: test_dict[key] for key in sorted(test_dict.keys(), key=lambda ele: ele[0] * ele[1])}",
    "Eczema commonly causes itchy, dry, inflamed patches of skin. The affected areas may look red, scaly, cracked, or darker than the surrounding skin depending on skin tone. Symptoms can flare after exposure to irritants, allergens, stress, or changes in weather.",
    "People with pollen allergy can reduce exposure by staying indoors on dry, windy days, avoiding early-morning outdoor activity, and going outside after rain when pollen levels are lower. They should check pollen forecasts, close windows and doors when counts are high, and consider starting allergy medication before symptoms begin if high pollen is expected. After being outside, showering, changing clothes, avoiding outdoor laundry drying, and wearing a face mask for yard work can help limit pollen contact.",
]
EXPECTED_DIAGONAL = [0.785, 0.651, 0.661, 0.799]
TOLERANCE = 0.03

results = []


def report(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


def call(base, key, path, payload=None, timeout=300):
    url = base.rstrip("/") + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method="POST" if payload is not None else "GET",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        return e.code, body[:600]
    except Exception as e:  # connection refused, DNS, timeout ...
        return None, str(e)[:600]


def chat(base, key, model, prompt, max_tokens):
    return call(base, key, "/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    })


def reasoning_of(msg):
    return msg.get("reasoning_content") or msg.get("reasoning") or ""


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(a):
    n = math.sqrt(dot(a, a))
    return [x / n for x in a] if n else a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="LiteLLM base URL incl. /v1")
    ap.add_argument("--key", required=True, help="LiteLLM key (same one Open WebUI uses)")
    ap.add_argument("--chat-model", default="qwen3.8-27b")
    ap.add_argument("--thinking-model", default="qwen3.8-27b-thinking", help="'none' to skip")
    ap.add_argument("--embed-model", default="nemotron-3-embed-8b")
    ap.add_argument("--rerank-model", default="bge-reranker-v2-m3", help="'none' to skip")
    ap.add_argument("--query-prefix", default="query: ")
    ap.add_argument("--passage-prefix", default="passage: ")
    ap.add_argument("--long-prompt-tokens", type=int, default=16000)
    a = ap.parse_args()

    # 1. models
    status, body = call(a.base, a.key, "/models")
    if status != 200:
        report("LiteLLM reachable", False, f"HTTP {status}: {body}")
        return
    names = {m.get("id") for m in body.get("data", [])}
    wanted = [a.chat_model, a.embed_model]
    wanted += [m for m in (a.thinking_model, a.rerank_model) if m != "none"]
    missing = [m for m in wanted if m not in names]
    report("models registered in LiteLLM", not missing,
           f"missing {missing}; LiteLLM has {sorted(n for n in names if n)}" if missing else "")

    # 2. chat, non-thinking
    status, body = chat(a.base, a.key, a.chat_model, "Reply with exactly the word READY.", 16)
    if status == 200:
        msg = body["choices"][0]["message"]
        content = msg.get("content") or ""
        thinking = reasoning_of(msg) or ("<think>" in content)
        report("chat answers", "READY" in content.upper(), repr(content[:80]))
        report("chat is non-thinking", not thinking,
               "model is still reasoning - check --default-chat-template-kwargs" if thinking else "")
    else:
        report("chat answers", False, f"HTTP {status}: {body}")

    # 3. typical RAG-sized prompt (" the" is a single token in Qwen's tokenizer)
    filler = "the " * a.long_prompt_tokens
    t0 = time.time()
    status, body = chat(a.base, a.key, a.chat_model, filler + "\nReply with OK.", 8)
    report(f"~{a.long_prompt_tokens}-token prompt accepted", status == 200,
           f"{time.time() - t0:.1f}s" if status == 200 else f"HTTP {status}: {body}")

    # 4. thinking alias
    if a.thinking_model != "none":
        status, body = chat(a.base, a.key, a.thinking_model, "What is 17 * 23?", 2048)
        if status == 200:
            msg = body["choices"][0]["message"]
            thinking = reasoning_of(msg) or "<think>" in (msg.get("content") or "")
            report("thinking alias reasons", bool(thinking),
                   "" if thinking else "no reasoning returned - request kwargs may not override the server default")
        else:
            report("thinking alias reasons", False, f"HTTP {status}: {body}")

    # 5. embeddings, same prefixing Open WebUI does
    texts = [a.query_prefix + q for q in QUERIES] + [a.passage_prefix + d for d in DOCUMENTS]
    status, body = call(a.base, a.key, "/embeddings", {"model": a.embed_model, "input": texts})
    if status != 200:
        report("embeddings", False, f"HTTP {status}: {body}")
    else:
        vecs = [norm(d["embedding"]) for d in sorted(body["data"], key=lambda d: d["index"])]
        dim = len(vecs[0])
        report("embedding dimension", dim == 4096, f"{dim}")
        q, d = vecs[:4], vecs[4:]
        scores = [[dot(qi, dj) for dj in d] for qi in q]
        print("       similarity matrix (rows=queries, cols=passages):")
        for i, row in enumerate(scores):
            print(f"       q{i} " + " ".join(f"{s:7.4f}" for s in row))
        diag_ok = all(abs(scores[i][i] - EXPECTED_DIAGONAL[i]) <= TOLERANCE for i in range(4))
        rank_ok = all(max(range(4), key=lambda j: row[j]) == i for i, row in enumerate(scores))
        report("each query ranks its own passage first", rank_ok)
        report("scores match NVIDIA model card (±0.03)", diag_ok,
               "" if diag_ok else f"expected diagonal ≈ {EXPECTED_DIAGONAL}: check pooling (must be MEAN) and prefixes")

    # 6. reranker
    if a.rerank_model != "none":
        status, body = call(a.base, a.key, "/rerank", {
            "model": a.rerank_model,
            "query": QUERIES[3],
            "documents": DOCUMENTS,
            "top_n": len(DOCUMENTS),
        })
        if status != 200:
            report("reranker", False, f"HTTP {status}: {body}")
        else:
            res = body.get("results", [])
            top = max(res, key=lambda r: r["relevance_score"]) if res else None
            report("reranker puts the relevant passage first", bool(top) and top["index"] == 3,
                   json.dumps([(r["index"], round(r["relevance_score"], 4)) for r in res]))


if __name__ == "__main__":
    main()
    passed = sum(results)
    print(f"\n{passed}/{len(results)} checks passed")
    sys.exit(0 if results and all(results) else 1)
