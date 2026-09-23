# RAG stack fix: Qwen3.8-27B + Nemotron-3-Embed-8B + LiteLLM + Open WebUI (airgapped)

```
Open WebUI ──► LiteLLM ──► vLLM qwen3.8-27b          (full GPU)
   │  (Tika)       ├─────► vLLM nemotron-3-embed-8b  (MIG 2g.48gb)
   │               └─────► vLLM bge-reranker-v2-m3   (optional, any spare slice)
   └── chunks, vectors (Chroma), BM25 index
```

| File | Layer |
|---|---|
| `01-vllm-qwen3.8-27b.values.yaml` | vLLM chat model |
| `02-vllm-nemotron-embed.values.yaml` | vLLM embedder |
| `03-vllm-reranker.values.yaml` | vLLM reranker (optional) |
| `04-litellm-proxy-config.yaml` | LiteLLM `proxy_config` |
| `05-open-webui.values.yaml` | Open WebUI `extraEnvVars` |
| `06-tika.values.yaml` | Tika subchart: no OCR, longer timeout, more CPU/RAM |
| `verify_rag_stack.py` | End-to-end test |

The vLLM files use the production-stack chart layout. If you use your own chart, copy the flags from `vllmConfig` and `extraArgs`. Service URLs are marked `# <-- your svc`.

---

## 1. Find which 400 you have

Look at **when** it fails, then read the log that owns that step.

| When | Message (Open WebUI / LiteLLM / vLLM log) | Cause | Fix |
|---|---|---|---|
| Chatting | `maximum context length is N tokens. However, you requested M` | Prompt + `max_tokens` > vLLM `--max-model-len`. Usually Full Context mode, a file set to "Entire Document", or a large Top K | 01: `maxModelLen: 131072`. 05: `RAG_FULL_CONTEXT=false`. Don't set Max Tokens above ~8k in model params |
| Chatting | LiteLLM `ContextWindowExceededError` | Same thing, as reported by LiteLLM | Same |
| Uploading | `400: 'NoneType' object has no attribute 'encode'` | No embedding model loaded. In an airgap this usually means the engine is still the local default and can't download | 05: `RAG_EMBEDDING_ENGINE=openai` plus the LiteLLM URL. Set it in the Admin UI too (step 5) |
| Uploading | `400: The content provided is empty` | Tika returned nothing: scanned PDF with no OCR, Tika timeout, or Tika OOM on 500 pages | Check Tika logs. Use the `-full` Tika image for OCR and give Tika 2–4 Gi of memory |
| Uploading | vLLM embedder `maximum context length is 8192` | A chunk is bigger than the embedder limit. Happens with the character splitter and huge CHUNK_SIZE | 05: token splitter, 800/100 |
| Hybrid search on | Download error / hang, then retrieval fails | Open WebUI tries to download the pre-filled reranker from Hugging Face | Section 4 |

Quick log commands:
```bash
kubectl logs deploy/<qwen-engine>  | grep -iE "context length|400"
kubectl logs deploy/<litellm>      | grep -iE "BadRequest|ContextWindow"
kubectl logs deploy/open-webui     | grep -iE "error|400|rerank|embedding"
```

---

## 2. Sizing a 500-page document

| | Value |
|---|---|
| Document | ~250–325k tokens. Bigger than Qwen's 262k native window, so full context can never work |
| Chunks | 800 tokens, overlap 100 → **~400–500 chunks** (earlier I estimated ~1,500; that was too high) |
| Ingestion | batch 32 → ~15 embedding calls. Tika extraction is usually the slow part |
| Per question | Top K 20 from BM25 + 20 from vectors → reranker → **top 8 chunks ≈ 7k tokens** to Qwen |
| Qwen request | ~7k context + ~1k template + chat history + answer ≈ 15–25k, far below 131k |

For documents this size, use a **Knowledge base** (Workspace → Knowledge). Attach it with `#` in chat, or bind it to a custom model. If you attach the PDF directly in a chat, click the file chip and make sure it says **Focused Retrieval**, not **Entire Document**.

---

## 3. Changes per layer

### vLLM: Qwen (`01`)
- `maxModelLen: 131072`. KV cache is cheap here because only 16 of 64 layers are full attention (~64 KiB/token in BF16).
- `--default-chat-template-kwargs '{"enable_thinking": false}'` makes non-thinking the server default, so nothing has to pass it per request.
- `--reasoning-parser qwen3`: when thinking is turned on, the reasoning comes back in its own field instead of as `<think>` text.
- `--override-generation-config`: Qwen's non-thinking sampling (temp 0.7, top_p 0.8, top_k 20). The checkpoint ships the thinking-mode defaults.
- `enablePrefixCaching: true`: pairs with `RAG_SYSTEM_CONTEXT=true`.
- Offline env: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`.

### vLLM: Nemotron embedder (`02`)
- `runner: pooling`, `maxModelLen: 8192`, served name `nemotron-3-embed-8b`.
- The OpenAI-style `/v1/embeddings` endpoint does **not** add NVIDIA's `query:`/`passage:` prefixes. Open WebUI adds them (layer 05).
- Pooling must be MEAN. The official checkpoint sets it, and `verify_rag_stack.py` checks it against NVIDIA's published scores.

### vLLM: reranker (`03`, optional)
- `BAAI/bge-reranker-v2-m3`, about 1.1 GB. It fits on any spare MIG slice.

### LiteLLM (`04`)
- `qwen3.8-27b`: non-thinking, used for RAG and for Open WebUI's background tasks.
- `qwen3.8-27b-thinking`: same backend, with thinking re-enabled through `extra_body.chat_template_kwargs`.
- `nemotron-3-embed-8b` (`mode: embedding`) and `bge-reranker-v2-m3` (`mode: rerank`).
- `api_base`: include `/v1` for chat and embeddings, leave it off for rerank. LiteLLM appends `/rerank` itself.
- `retry_policy.BadRequestErrorRetries: 0` stops a context-length 400 from being retried and multiplied.

### Open WebUI (`05`): what each key does

| Setting | Value | Why |
|---|---|---|
| `RAG_EMBEDDING_ENGINE` / `RAG_OPENAI_API_BASE_URL` / `RAG_EMBEDDING_MODEL` | `openai` / LiteLLM `/v1` / `nemotron-3-embed-8b` | Embeds through LiteLLM |
| `RAG_EMBEDDING_QUERY_PREFIX` | `"query: "` | Prepended to every search query before embedding. **Env-only**, restart needed |
| `RAG_EMBEDDING_CONTENT_PREFIX` | `"passage: "` | Prepended to every chunk before embedding. **Env-only**, restart needed |
| `RAG_SYSTEM_CONTEXT` | `true` | The "set to true" one. Retrieved chunks go in the system message at a fixed position, so vLLM reuses its prefix cache and follow-ups answer much faster. **Env-only** |
| `RAG_TEXT_SPLITTER` / `CHUNK_SIZE` / `CHUNK_OVERLAP` | `token` / `800` / `100` | Token-based. The default splitter counts characters |
| `RAG_EMBEDDING_BATCH_SIZE` | `32` | The default of 1 means one HTTP call per chunk |
| `ENABLE_RAG_HYBRID_SEARCH` | `true` | BM25 + vectors. Helps with exact terms, IDs and clause numbers |
| `RAG_TOP_K` / `RAG_TOP_K_RERANKER` | `20` / `8` | Wide candidate pool, narrow final context |
| `RAG_FULL_CONTEXT` / `BYPASS_EMBEDDING_AND_RETRIEVAL` | `false` / `false` | Never send the whole document |
| `TASK_MODEL_EXTERNAL` | `qwen3.8-27b` | Query generation and titles without `<think>` output |
| `OFFLINE_MODE`, `HF_HUB_OFFLINE`, `*_AUTO_UPDATE=false` | airgap | No download attempts |

---

## 4. Hybrid search in an airgap: the reranker

When you tick Hybrid Search, the reranking field is pre-filled with a Hugging Face model, and Open WebUI tries to download it locally. Pick one of these:

- **Option A: a GPU reranker (better answers).** Deploy `03`, add it to LiteLLM, and set in Open WebUI: Reranking Engine = **External**, URL = `http://<litellm>:4000/v1/rerank` (the full path; Open WebUI appends nothing), Model = `bge-reranker-v2-m3`. With the engine set to external, Open WebUI never loads or downloads a local model.
- **Option B: no spare GPU.** Keep Hybrid Search on and **clear the Reranking Model field** (engine left empty), then Save. Open WebUI's code falls back to cosine scoring with your Nemotron embedder. It needs no download and no extra pod. Use Top K 10 / Top K Reranker 6, because the candidates are re-embedded on every query.
- Also possible: pre-seed a CrossEncoder into `/app/backend/data/cache/embedding/models/` on the PVC and run it on CPU. It works, but expect seconds per query.

---

## 5. Rollout order

1. **vLLM**: apply `01`, `02`, and `03` if you use it. Check `curl http://<svc>:8000/v1/models`. The Qwen entry should show `max_model_len: 131072`.
2. **LiteLLM**: apply `04` and restart. `curl -H "Authorization: Bearer $KEY" http://<litellm>:4000/v1/models` should list all 3–4 names.
3. **Open WebUI Helm**: apply `05`. The env-only vars take effect on this restart.
4. **Existing install: also set the values in the Admin UI.** RAG settings are persistent config, so the value already saved in the database overrides Helm. Go to **Admin Panel → Settings → Documents** and set: Content Extraction = Tika; Text Splitter = Token; Chunk 800 / Overlap 100; Embedding engine = OpenAI + LiteLLM URL/key; model `nemotron-3-embed-8b`; batch 32; Full Context off; Bypass off; Hybrid on; the reranker from section 4; Top K 20; Top K Reranker 8; Relevance threshold 0. **Save.**
   - Alternative: set `RESET_CONFIG_ON_START=true` for one restart, then remove it. This resets **all** persisted settings to env and defaults, including connections you only created in the UI. Use it only if everything you need is in Helm.
5. **Re-index.** A new embedding model, new prefixes and new chunking make the old vectors unusable. Use **Admin → Settings → Documents → Reindex Knowledge Base Vectors**, and re-upload any files that aren't in a knowledge base.
6. **Model settings** (Admin → Models → qwen3.8-27b → Advanced Params): leave Max Tokens unset or at ≤ 8192. Keep Function Calling = Default for classic RAG. Native mode switches to agentic retrieval, where Qwen must call a search tool itself. Optionally set presence_penalty 1.5 (Qwen's non-thinking recommendation) if you see repetition.
7. **Hide the embedder and reranker from the chat picker** (Admin → Models → toggle off). Choosing one of them as the chat model also fails, often with a 400.
8. **Verify**:
   ```bash
   kubectl exec -i deploy/open-webui -- python3 - \
     --base http://litellm.litellm.svc.cluster.local:4000/v1 --key "$LITELLM_KEY" \
     < verify_rag_stack.py          # add --rerank-model none for Option B
   ```
   All checks should pass. The embedding check compares your similarity scores with NVIDIA's model card, which catches wrong pooling or missing prefixes.

---

## 6. Slow uploads to a knowledge base

**First find the slow step** from Open WebUI's own log lines while you upload:
```bash
kubectl logs -f deploy/open-webui | grep -E "generating embeddings|embeddings generated|added .* items"
```
- **Long gap between the upload and the first `generating embeddings`** means Tika is slow. That's almost always OCR. Use `06-tika.values.yaml`.
- **Long gap between `generating embeddings` and `embeddings generated`** means embedding is slow. Use batch 32 and concurrency 4 (`05`). A knowledge-base add embeds the chunks **twice**: once for the file and once for the knowledge base. With the default batch size of 1, that's thousands of HTTP calls.
- **Long gap between `embeddings generated` and `added … items`** means the vector store is slow. Chroma is SQLite on the Open WebUI PVC; put that PVC on block storage, not NFS or CephFS.

**Clean way to load a big PDF:**
1. Check it has a text layer (you can select text in a PDF viewer). If it's scanned, OCR it once offline (`ocrmypdf --jobs 8 in.pdf out.pdf`) and upload the result.
2. Run Tika without OCR (`06`) and with `PDF_EXTRACT_IMAGES=false` (`05`).
3. Upload once, straight into the knowledge base (Workspace → Knowledge → your KB → +). For many files, use the API: upload the file, poll `/api/v1/files/{id}/process/status` until it reports `completed`, then add it to the knowledge base.

---

## 7. Airgap checklist

- [ ] Model weights for all vLLM models are copied into the PVCs, and `modelURL` points to those local paths.
- [ ] The Open WebUI image is the standard one, not `useSlim`. It contains the tiktoken `cl100k_base` file that the token splitter needs, and the chart's `copy-app-data` init container copies it onto the PVC. Check with `kubectl exec deploy/open-webui -- ls /app/backend/data/cache/tiktoken`.
- [ ] If you use Tika for OCR, the Tika image is `-full`, mirrored.
- [ ] If `VECTOR_DB=pgvector`: 4096-dim vectors can't be indexed (the `vector` type caps at 2000 dims, `halfvec` indexing at 4000). The default Chroma has no such limit.
