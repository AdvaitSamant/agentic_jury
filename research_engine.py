import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODEL = "nvidia/nemotron-3.5-super-vl:free"
MEMORY_PATH = Path(__file__).with_name("research_memory.json")

LOCAL_RESEARCH_DOCUMENTS = [
    {
        "source": "LOCAL-QC-1",
        "title": "Quantum computing basics (local sample)",
        "text": (
            "Quantum computers use qubits. Unlike a classical bit, a qubit can "
            "be prepared in a superposition of basis states. Quantum gates "
            "transform qubit states, and measurement produces classical outcomes."
        ),
    },
    {
        "source": "LOCAL-QC-2",
        "title": "Quantum error correction (local sample)",
        "text": (
            "Quantum states are fragile and errors can arise from noise and "
            "interactions with the environment. Error-correction methods encode "
            "logical information across physical qubits to detect and correct "
            "errors. Fault-tolerant computation remains a challenge."
        ),
    },
    {
        "source": "LOCAL-QC-3",
        "title": "Applications and limitations (local sample)",
        "text": (
            "Quantum computing is researched for problems in chemistry, "
            "materials, and optimization. Practical advantage depends on the "
            "problem and hardware. Current devices have limitations including "
            "noise, limited scale, and the need for reliable error correction."
        ),
    },
    {
        "source": "LOCAL-QC-4",
        "title": "Quantum algorithms (local sample)",
        "text": (
            "Quantum algorithms use quantum operations for specific "
            "computational tasks. A quantum approach is not automatically "
            "faster for every problem; performance should be compared with "
            "suitable classical methods."
        ),
    },
]


@dataclass(frozen=True)
class WorkflowSettings:
    online_llm: bool = True
    web_search: bool = True
    openrouter_api_key: str | None = None
    serper_api_key: str | None = None
    model: str = MODEL
    memory_path: Path = MEMORY_PATH


@dataclass
class _RunContext:
    settings: WorkflowSettings
    llm_active: bool
    llm_mode: str
    llm_status_detail: str
    search_modes: set[str] = field(default_factory=set)


def _new_run_context(settings: WorkflowSettings) -> _RunContext:
    if not settings.online_llm:
        detail = "Online LLM requests are disabled."
    elif not settings.openrouter_api_key:
        detail = "OpenRouter key is missing; deterministic local logic is active."
    else:
        detail = "OpenRouter will be tested during the research run."

    return _RunContext(
        settings=settings,
        llm_active=settings.online_llm and bool(settings.openrouter_api_key),
        llm_mode="OFFLINE DEMO MODE",
        llm_status_detail=detail,
    )


def _openrouter_chat(
    settings: WorkflowSettings,
    messages: list[dict[str, Any]],
    temperature: float = 0.3,
    max_tokens: int = 1200,
) -> str:
    if not settings.openrouter_api_key:
        raise RuntimeError("OpenRouter API key is missing.")

    try:
        import requests
    except ImportError:
        raise RuntimeError("The requests package is unavailable.") from None

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "reasoning": {"enabled": True},
            },
            timeout=45,
        )
    except requests.RequestException as error:
        raise RuntimeError(
            f"OpenRouter network request failed ({type(error).__name__})."
        ) from None

    if response.status_code >= 400:
        raise RuntimeError(f"OpenRouter returned HTTP {response.status_code}.")

    try:
        payload = response.json()
    except ValueError:
        raise RuntimeError(
            "OpenRouter returned an unexpected response format."
        ) from None

    def get_message(response_payload: dict[str, Any] | None) -> dict[str, Any] | None:
        choices = response_payload.get("choices") if response_payload else None
        if not isinstance(choices, list) or not choices:
            return None
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        return message if isinstance(message, dict) else None

    first_message = get_message(payload if isinstance(payload, dict) else None)
    if first_message is None:
        raise RuntimeError("OpenRouter response did not include an assistant message.")

    reasoning_details = first_message.get("reasoning_details")
    if reasoning_details is not None:
        continuation_messages = [
            *messages,
            {
                "role": "assistant",
                "content": first_message.get("content"),
                "reasoning_details": reasoning_details,
            },
            {"role": "user", "content": "Are you sure? Think carefully."},
        ]
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.openrouter_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.model,
                    "messages": continuation_messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "reasoning": {"enabled": True},
                },
                timeout=45,
            )
        except requests.RequestException as error:
            raise RuntimeError(
                f"OpenRouter network request failed ({type(error).__name__})."
            ) from None

        if response.status_code >= 400:
            raise RuntimeError(
                f"OpenRouter returned HTTP {response.status_code}."
            )
        try:
            payload = response.json()
        except ValueError:
            raise RuntimeError(
                "OpenRouter returned an unexpected response format."
            ) from None
        first_message = get_message(payload if isinstance(payload, dict) else None)

    text = first_message.get("content") if first_message is not None else None
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("OpenRouter response did not include assistant text.")
    return text.strip()


def _word_set(text: str) -> set[str]:
    ignored = {
        "about",
        "after",
        "also",
        "and",
        "are",
        "for",
        "from",
        "how",
        "into",
        "its",
        "latest",
        "more",
        "research",
        "that",
        "the",
        "their",
        "this",
        "what",
        "when",
        "where",
        "which",
        "with",
        "would",
        "does",
        "can",
        "could",
        "should",
        "explain",
        "describe",
        "find",
        "using",
        "use",
        "current",
        "potential",
        "overview",
        "relevant",
        "task",
        "tasks",
        "concepts",
        "capabilities",
        "applications",
        "benefits",
        "limitations",
        "questions",
        "open",
        "evidence",
    }
    return {
        word
        for word in re.findall(r"[a-z0-9]+", text.lower())
        if len(word) > 2 and word not in ignored
    }


def _local_search(
    query: str, context: _RunContext, limit: int = 5
) -> list[dict[str, Any]]:
    context.search_modes.add("offline local sample")
    terms = _word_set(query)
    ranked = []
    for document in LOCAL_RESEARCH_DOCUMENTS:
        score = len(terms & _word_set(document["title"] + " " + document["text"]))
        if score:
            ranked.append((score, document))
    ranked.sort(key=lambda pair: (-pair[0], pair[1]["source"]))
    return [
        dict(document, score=score, mode="offline local sample")
        for score, document in ranked[:limit]
    ]


def _serper_search(
    query: str, context: _RunContext, limit: int = 5
) -> list[dict[str, Any]] | None:
    settings = context.settings
    if not (
        settings.web_search
        and settings.serper_api_key
    ):
        return None

    try:
        import requests
    except ImportError:
        return None

    try:
        response = requests.post(
            "https://google.serper.dev/search",
            headers={
                "X-API-KEY": settings.serper_api_key,
                "Content-Type": "application/json",
            },
            json={"q": query, "num": limit},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        organic_results = (
            payload.get("organic", []) if isinstance(payload, dict) else []
        )
        if not isinstance(organic_results, list):
            organic_results = []
        results = [
            {
                "source": item.get("link", "Unknown URL"),
                "title": item.get("title", "Untitled result"),
                "text": item.get("snippet", ""),
                "mode": "live Serper search",
            }
            for item in organic_results[:limit]
            if isinstance(item, dict) and item.get("snippet")
        ]
    except (requests.RequestException, ValueError):
        return None

    if results:
        context.search_modes.add("live Serper search")
        return results
    return None


def collect_sources(
    query: str, context: _RunContext, limit: int = 5
) -> list[dict[str, Any]]:
    live_results = _serper_search(query, context, limit)
    if live_results:
        return live_results
    return _local_search(query, context, limit)


def web_search(query: str, context: _RunContext) -> str:
    results = collect_sources(query, context)
    if not results:
        return f"No matching source was found for: {query}"
    return "\n\n".join(
        f"[{item['mode']}] {item['title']}\n"
        f"Source: {item['source']}\n"
        f"Snippet: {item['text']}"
        for item in results
    )


def call_llm(
    context: _RunContext,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
) -> str:
    if not context.llm_active:
        raise RuntimeError("Online LLM mode is not active.")
    try:
        result = _openrouter_chat(
            context.settings,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
    except RuntimeError as error:
        context.llm_active = False
        context.llm_mode = "OFFLINE DEMO MODE"
        context.llm_status_detail = (
            f"{error} Deterministic local logic is active."
        )
        raise RuntimeError(
            "OpenRouter request failed; using offline fallback."
        ) from None

    context.llm_mode = "ONLINE LLM MODE"
    context.llm_status_detail = "OpenRouter returned a response for this run."
    return result


def plan_subquestions(question: str, context: _RunContext) -> list[str]:
    if context.llm_active:
        try:
            reply = call_llm(
                context,
                "You are a research planner. Return 3 to 5 numbered focused tasks.",
                f"Decompose this research goal into focused tasks: {question}",
            )
            tasks = [
                match.group(1).strip()
                for line in reply.splitlines()
                if (match := re.match(r"^\s*\d+[.)]\s*(.+)", line))
            ]
            if tasks:
                return tasks[:5]
        except RuntimeError:
            pass

    return [
        f"Explain core concepts and capabilities of: {question}",
        f"Find applications and possible benefits of: {question}",
        f"Find limitations and open questions about: {question}",
    ]


def research_subquestion(subquestion: str, context: _RunContext) -> str:
    results = collect_sources(subquestion, context)
    if not results:
        return "No relevant information was found in the available sources."

    raw = "\n\n".join(
        f"Title: {item['title']}\n"
        f"Source: {item['source']}\n"
        f"Finding: {item['text']}"
        for item in results
    )
    if context.llm_active:
        try:
            return call_llm(
                context,
                "Filter provided results only. Keep source labels and do not invent facts.",
                f"Question: {subquestion}\n\nResults:\n{raw}\n\nSummarize relevant evidence.",
            )
        except RuntimeError:
            pass
    return "[OFFLINE DETERMINISTIC SOURCE EXCERPTS]\n" + raw


def write_report(
    topic: str, subquestions: list[str], findings: dict[str, str], context: _RunContext
) -> str:
    combined = "\n\n".join(
        f"Task: {question}\n{findings.get(question, 'No evidence found.')}"
        for question in subquestions
    )
    if context.llm_active:
        try:
            return call_llm(
                context,
                "Write a concise cited research report from supplied findings only. Mark gaps.",
                f"Topic: {topic}\n\nFindings:\n{combined}\n\n"
                "Use Overview, Evidence, Limitations, Sources headings.",
                temperature=0.4,
            )
        except RuntimeError:
            pass

    evidence = "\n".join(
        f"- **{question}**\n  {findings[question]}"
        for question in subquestions
        if findings.get(question)
        and "No relevant information" not in findings[question]
    )
    if not evidence:
        evidence = "No supporting source text was retrieved."
    return (
        f"# Research report: {topic}\n\n"
        "## Overview\n"
        "Offline report assembled deterministically from available excerpts; "
        "it was not generated by an LLM.\n\n"
        f"## Evidence\n{evidence}\n\n"
        "## Limitations / Open Questions\n"
        "Coverage is limited to the available source snippets; this is not a "
        "comprehensive literature review.\n\n"
        "## Sources\nSource labels are shown beside each finding."
    )


def refine_report(
    topic: str,
    subquestions: list[str],
    draft_report: str,
    context: _RunContext,
) -> str:
    if context.llm_active:
        try:
            tasks = "\n".join(f"- {question}" for question in subquestions)
            return call_llm(
                context,
                "Check report coverage against tasks. Do not add facts or sources.",
                f"Topic: {topic}\nTasks:\n{tasks}\n\n"
                f"Draft:\n{draft_report}\n\nReturn refined report.",
                temperature=0.3,
            )
        except RuntimeError:
            pass

    missing = [
        question
        for question in subquestions
        if question.casefold() not in draft_report.casefold()
    ]
    if missing:
        return (
            draft_report
            + "\n\n## Refinement note\nCoverage could not be confirmed for: "
            + "; ".join(missing)
        )
    return draft_report


def run_original_research_assistant(
    topic: str, settings: WorkflowSettings | None = None
) -> str:
    context = _new_run_context(settings or WorkflowSettings())
    questions = plan_subquestions(topic, context)
    findings = {
        question: research_subquestion(question, context)
        for question in questions
    }
    draft = write_report(topic, questions, findings, context)
    return refine_report(topic, questions, draft, context)


def planner_agent(question: str, context: _RunContext) -> dict[str, Any]:
    return {
        "role": "Planner",
        "question": question,
        "tasks": plan_subquestions(question, context),
    }


def _source_matches_task(task: str, source: dict[str, Any]) -> bool:
    task_terms = _word_set(task)
    source_text = " ".join(
        str(source.get(key, ""))
        for key in ("title", "text", "evidence", "claim")
    )
    return bool(task_terms & _word_set(source_text))


def researcher_agent(
    plan: dict[str, Any], context: _RunContext
) -> dict[str, Any]:
    findings = []
    for task in plan["tasks"]:
        for source in collect_sources(task, context, limit=4):
            if not _source_matches_task(task, source):
                continue
            findings.append(
                {
                    "task": task,
                    "claim": source["text"],
                    "source": source["source"],
                    "title": source["title"],
                    "evidence": source["text"],
                    "mode": source["mode"],
                }
            )
    return {"role": "Researcher", "findings": findings}


def verifier_agent(
    plan: dict[str, Any],
    research: dict[str, Any],
    extra_claims: list[str] | None = None,
) -> dict[str, Any]:
    supported = []
    unsupported = []
    covered = set()
    seen_evidence = set()
    planned_tasks = set(plan["tasks"])

    for item in research["findings"]:
        task = item.get("task", "")
        has_evidence = bool(item.get("evidence", "").strip())
        has_source = bool(item.get("source", "").strip())

        if task not in planned_tasks:
            unsupported.append(
                {
                    "claim": item.get("claim", ""),
                    "reason": "This finding does not match a Planner task.",
                }
            )
        elif not has_evidence or not has_source:
            unsupported.append(
                {
                    "claim": item.get("claim", ""),
                    "reason": "Missing source-backed evidence.",
                }
            )
        elif not _source_matches_task(task, item):
            unsupported.append(
                {
                    "claim": item.get("claim", ""),
                    "reason": (
                        "Source text has no relevant task keywords; "
                        "not counted as support."
                    ),
                }
            )
        else:
            covered.add(task)
            evidence_key = hashlib.sha256(
                f"{item.get('claim', '')}|{item.get('source', '')}".encode()
            ).hexdigest()[:16]
            if evidence_key not in seen_evidence:
                supported.append(item)
                seen_evidence.add(evidence_key)

    for claim in extra_claims or []:
        unsupported.append(
            {
                "claim": claim,
                "reason": (
                    "Not supported by the research results; excluded from report."
                ),
            }
        )

    missing = [task for task in plan["tasks"] if task not in covered]
    return {
        "role": "Verifier",
        "verified_findings": supported,
        "unsupported_claims": unsupported,
        "missing_tasks": missing,
    }


def synthesizer_agent(
    plan: dict[str, Any], verification: dict[str, Any]
) -> str:
    verified = verification["verified_findings"]
    evidence = "\n".join(
        f"- {item['claim']} (source: {item['source']}; {item['title']})"
        for item in verified
    ) or "No supported findings were available for this question."
    unsupported = "\n".join(
        f"- {item['claim']} — {item['reason']}"
        for item in verification["unsupported_claims"]
    ) or "No unsupported claims were provided."
    missing = "\n".join(
        f"- {task}" for task in verification["missing_tasks"]
    ) or "No planned task was left without a matching source."

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


def run_multi_agent_research(
    question: str,
    context: _RunContext,
    extra_claims: list[str] | None = None,
) -> dict[str, Any]:
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


def load_memories(path: Path = MEMORY_PATH) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(
            f"Cannot read memory JSON ({error.__class__.__name__})."
        ) from None
    if not isinstance(records, list):
        raise ValueError("Memory JSON must contain a list.")
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("Every memory entry must be a JSON object.")
    return records


def save_memory(
    topic: str,
    content: str,
    source: str | None = None,
    path: Path = MEMORY_PATH,
) -> str:
    path = Path(path)
    records = load_memories(path)
    memory_id = hashlib.sha256(
        f"{topic}|{content}|{source}".encode()
    ).hexdigest()[:16]
    if any(record.get("id") == memory_id for record in records):
        return memory_id

    records.append(
        {
            "id": memory_id,
            "topic": topic,
            "content": content,
            "source": source,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return memory_id


def retrieve_memories(
    query: str, path: Path = MEMORY_PATH, limit: int = 5
) -> list[dict[str, Any]]:
    records = load_memories(path)
    terms = _word_set(query)
    ranked = []
    for record in records:
        text = " ".join(
            str(record.get(key, "")) for key in ("topic", "content", "source")
        )
        overlap = len(terms & _word_set(text))
        if overlap:
            ranked.append((overlap, record))
    ranked.sort(
        key=lambda pair: (
            -pair[0],
            pair[1].get("saved_at_utc", ""),
        )
    )
    return [record for _, record in ranked[:limit]]


def run_integrated_workflow(
    question: str,
    settings: WorkflowSettings | None = None,
    save_findings: bool = True,
) -> dict[str, Any]:
    if not question.strip():
        raise ValueError("Enter a research question before starting a run.")

    settings = settings or WorkflowSettings()
    context = _new_run_context(settings)
    result = run_multi_agent_research(question.strip(), context)

    saved_memory_ids = set()
    if save_findings:
        for finding in result["verification"]["verified_findings"]:
            saved_memory_ids.add(
                save_memory(
                    question.strip(),
                    finding["claim"],
                    finding["source"],
                    settings.memory_path,
                )
            )

    result["run_status"] = {
        "llm_mode": context.llm_mode,
        "llm_status_detail": context.llm_status_detail,
        "source_modes": sorted(context.search_modes),
    }
    result["memory_saved_count"] = len(saved_memory_ids)
    return result
