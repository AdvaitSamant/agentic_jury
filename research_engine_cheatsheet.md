# `research_engine.py` — Presentation Cheatsheet

## What the engine does

It turns a research question into smaller tasks, gathers source snippets for
those tasks, checks that findings are relevant and source-backed, and assembles
a report. When enabled, it can use OpenRouter for language-model work and
Serper for live web search. Without those services, it has deterministic local
fallbacks.

### One-slide workflow

```text
Question
   ↓
Planner: split into focused tasks
   ↓
Researcher: collect relevant source snippets
   ↓
Verifier: separate supported findings from unsupported claims
   ↓
Synthesizer: produce a report, gaps, and limitations
   ↓
Optional: save verified findings to JSON memory
```

The public entry point is `run_integrated_workflow`; the core multi-agent
pipeline is `run_multi_agent_research`.

---

## 1. Settings and per-run state

```python
@dataclass(frozen=True)
class WorkflowSettings:
    online_llm: bool = True
    web_search: bool = True
    model: str = MODEL
    memory_path: Path = MEMORY_PATH
```

**What it does:** Groups the service switches, model name, and memory-file
location. A run context tracks whether the LLM is active and which search
modes actually returned sources.

---

## 2. Live search with local fallback

```python
def collect_sources(query, context, limit=5):
    live_results = _serper_search(query, context, limit)
    if live_results:
        return live_results
    return _local_search(query, context, limit)
```

**What it does:** Attempts Serper search first. If live search is disabled,
unconfigured, unavailable, or produces no usable results, it searches the
built-in local sample documents instead.

Local search tokenizes the query and source text, scores documents by shared
keywords, and returns the highest-scoring matches. The local sample content is
illustrative, not a general-purpose or comprehensive source collection.

---

## 3. Planning: turn a topic into tasks

```python
def planner_agent(question, context):
    return {
        "role": "Planner",
        "question": question,
        "tasks": plan_subquestions(question, context),
    }
```

**What it does:** Builds the first pipeline artifact: a list of research tasks.
When the LLM is active, `plan_subquestions` asks it for 3–5 focused tasks. If
not, it creates three deterministic tasks covering concepts, applications,
and limitations.

---

## 4. Research: gather task-linked findings

```python
def researcher_agent(plan, context):
    findings = []
    for task in plan["tasks"]:
        for source in collect_sources(task, context, limit=4):
            if not _source_matches_task(task, source):
                continue
            findings.append({
                "task": task,
                "claim": source["text"],
                "source": source["source"],
                "title": source["title"],
                "evidence": source["text"],
                "mode": source["mode"],
            })
    return {"role": "Researcher", "findings": findings}
```

**What it does:** Searches for each planned task, filters out sources with no
keyword overlap, and records the source text and its provenance together.
Keeping the task, evidence, title, source, and search mode together makes each
finding traceable through later stages.

---

## 5. Verification: check evidence and coverage

```python
if task not in planned_tasks:
    unsupported.append({
        "claim": item.get("claim", ""),
        "reason": "This finding does not match a Planner task.",
    })
elif not has_evidence or not has_source:
    unsupported.append({
        "claim": item.get("claim", ""),
        "reason": "Missing source-backed evidence.",
    })
elif not _source_matches_task(task, item):
    unsupported.append({
        "claim": item.get("claim", ""),
        "reason": "Source text has no relevant task keywords; not counted as support.",
    })
else:
    covered.add(task)
    supported.append(item)
```

**What it does:** Accepts findings only when they map to a planned task and
include relevant evidence and a source. It also tracks which tasks have
supporting findings, removes duplicate evidence, and reports unsupported claims
and tasks with no matching source.

**Important limitation:** Relevance is measured using keyword overlap
(`_source_matches_task`), not semantic reasoning or fact-checking. A “verified”
finding has passed these code checks; that does not independently prove the
source or claim is true.

---

## 6. Synthesis: make the report

```python
def synthesizer_agent(plan, verification):
    verified = verification["verified_findings"]
    # Format verified evidence, unsupported claims, and missing tasks.
    return (
        f"# Research report: {plan['question']}\n\n"
        f"## Verified evidence\n{evidence}\n\n"
        f"## Unsupported claims\n{unsupported}\n\n"
        f"## Missing information\n{missing}\n\n"
        "## Evidence limitations\n"
        "Relevance is checked with keyword overlap, not semantic reasoning. "
        "Source references show where snippets came from but do not "
        "independently prove their truth."
    )
```

**What it does:** Produces a report that keeps supported evidence, rejected
claims, and research gaps visible rather than silently mixing them together.
This stage is deterministic; the separate legacy workflow functions
(`write_report` and `refine_report`) can use the LLM and provide offline
fallbacks.

---

## 7. Pipeline orchestration

```python
def run_multi_agent_research(question, context, extra_claims=None):
    plan = planner_agent(question, context)
    research = researcher_agent(plan, context)
    verification = verifier_agent(plan, research, extra_claims)
    report = synthesizer_agent(plan, verification)
    return {
        "plan": plan,
        "research": research,
        "verification": verification,
        "report": report,
    }
```

**What it does:** Runs the stages in order and returns each intermediate
artifact as well as the final report. This makes the plan, raw findings, and
verification results inspectable—not just the final text.

---

## 8. Memory: save and retrieve findings

```python
memory_id = hashlib.sha256(
    f"{topic}|{content}|{source}".encode()
).hexdigest()[:16]
if any(record.get("id") == memory_id for record in records):
    return memory_id
```

**What it does:** Creates a stable ID from a finding’s topic, content, and
source. Existing IDs are not appended again, so repeating the same run does
not create duplicate records. Records are stored in `research_memory.json`;
`retrieve_memories` ranks them by keyword overlap with a query.

---

## 9. The entry point: validate, run, and optionally save

```python
def run_integrated_workflow(question, settings=None, save_findings=True):
    if not question.strip():
        raise ValueError("Enter a research question before starting a run.")

    settings = settings or WorkflowSettings()
    context = _new_run_context(settings)
    result = run_multi_agent_research(question.strip(), context)

    if save_findings:
        for finding in result["verification"]["verified_findings"]:
            save_memory(
                question.strip(),
                finding["claim"],
                finding["source"],
                settings.memory_path,
            )
```

**What it does:** Rejects blank questions, creates default settings when
needed, runs the pipeline, and optionally persists only verified findings.
The returned result also includes the run mode, source modes, and count of
unique saved memories.

---

## 10. OpenRouter and Serper integrations

OpenRouter receives chat messages and generation settings:

```python
{
    "model": settings.model,
    "messages": messages,
    "temperature": temperature,
    "max_tokens": max_tokens,
    "reasoning": {"enabled": True},
}
```

The engine checks HTTP and response structure, then requires non-empty
assistant text. If the response includes `reasoning_details`, it makes a
follow-up request asking the model to think carefully before returning the
final text.

Serper's organic results are normalized into a common source shape:

```python
{
    "source": item.get("link", "Unknown URL"),
    "title": item.get("title", "Untitled result"),
    "text": item.get("snippet", ""),
    "mode": "live Serper search",
}
```

**Why it matters:** Both online and local results use the same fields, so the
Researcher can process them through the same pipeline.

---

## 11. Keyword matching and source ranking

```python
def _source_matches_task(task, source):
    task_terms = _word_set(task)
    source_text = " ".join(
        str(source.get(key, ""))
        for key in ("title", "text", "evidence", "claim")
    )
    return bool(task_terms & _word_set(source_text))
```

`_word_set` lowercases text, extracts alphanumeric words, and removes short
tokens and a stop-word list. Local search scores a document by the number of
shared words, then sorts by score and source ID for stable ordering.

**Trade-off:** This is simple and explainable, but it can miss relevant
paraphrases or match words without understanding their meaning.

---

## 12. Verification: duplicate handling and injected claims

```python
evidence_key = hashlib.sha256(
    f"{item.get('claim', '')}|{item.get('source', '')}".encode()
).hexdigest()[:16]
if evidence_key not in seen_evidence:
    supported.append(item)
    seen_evidence.add(evidence_key)
```

The Verifier deduplicates by claim + source, while task coverage is recorded
when a finding passes the support checks. Optional `extra_claims` are always
listed as unsupported because they were not produced by the source-backed
research results.

---

## 13. Memory file behavior

```python
records = json.loads(path.read_text(encoding="utf-8"))
if not isinstance(records, list):
    raise ValueError("Memory JSON must contain a list.")
if any(not isinstance(record, dict) for record in records):
    raise ValueError("Every memory entry must be a JSON object.")
```

`load_memories` treats a missing file as an empty memory store, but reports
malformed JSON or an invalid record shape as an error. Saved entries include a
UTC timestamp. `retrieve_memories` ranks saved records by overlapping topic,
content, and source keywords; memory retrieval is available as a helper and is
not automatically added to the research pipeline.

---

## 14. There are two workflow paths

`run_integrated_workflow` uses the Planner/Researcher/Verifier/Synthesizer
pipeline described above. The file also retains an earlier linear workflow:

```python
questions = plan_subquestions(topic, context)
findings = {
    question: research_subquestion(question, context)
    for question in questions
}
draft = write_report(topic, questions, findings, context)
return refine_report(topic, questions, draft, context)
```

That path can use the LLM to summarize source snippets, draft a report, and
check task coverage. The current multi-agent path uses the LLM for planning
when available; its research is based on retrieved snippets, and its
Synthesizer is deterministic. This distinction can help explain why the two
paths may produce reports in different formats.

---

## LLM and fallback behavior

- `call_llm` sends prompts to OpenRouter when online LLM use is enabled and
  configured.
- In the current multi-agent path, the LLM is used for planning; report
  synthesis remains deterministic.
- An OpenRouter request failure switches the run context to offline mode and
  lets the caller use deterministic fallback behavior.
- If live search cannot return results, `collect_sources` falls back to local
  sample documents.
- `run_status` reports the LLM mode and the source modes used in that run.

## Suggested presentation summary

> “The engine decomposes a research question into tasks, gathers source-backed
> snippets, checks task relevance and coverage, then creates a report that
> exposes unsupported claims and gaps. Online services are optional; local
> fallbacks keep the workflow runnable. Its verification is a lightweight
> keyword check, so citations and findings still need human review.”
