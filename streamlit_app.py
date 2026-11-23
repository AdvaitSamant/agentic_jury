import os
import re
from datetime import datetime

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from research_engine import MEMORY_PATH, WorkflowSettings


DEFAULT_TOPIC = "The latest developments in quantum computing"

st.set_page_config(
    page_title="Research desk",
    page_icon=":material/description:",
    layout="wide",
)




def configured_secret(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value

    try:
        value = st.secrets.get(name)
    except StreamlitSecretNotFoundError:
        return None
    return str(value) if value else None


def make_report_filename(topic: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")
    return f"{slug[:60] or 'research-brief'}.md"


openrouter_api_key = configured_secret("OPENROUTER_API_KEY")
serper_api_key = configured_secret("SERPER_API_KEY")

with st.sidebar:
    st.markdown("#### Research desk")
    st.caption("A focused workspace for sourced research")

    st.subheader("Services")
    use_online_llm = st.toggle(
        "Request OpenRouter",
        value=True,
        help="Uses the configured OpenRouter key; falls back to local logic if unavailable.",
    )
    use_live_search = st.toggle(
        "Use live web search",
        value=True,
        help="Uses Serper when configured; otherwise searches the local sample documents.",
    )
    save_findings = st.toggle(
        "Save verified findings",
        value=True,
        help="Stores verified findings in a local JSON file for later retrieval.",
    )

    if use_online_llm and openrouter_api_key:
        st.badge(
            "OpenRouter key detected",
            icon=":material/key:",
            color="green",
        )
    elif use_online_llm:
        st.badge(
            "Deterministic fallback",
            icon=":material/offline_bolt:",
            color="orange",
        )
    else:
        st.badge(
            "Offline mode",
            icon=":material/offline_bolt:",
            color="gray",
        )

    if use_live_search and serper_api_key:
        st.badge(
            "Serper key detected",
            icon=":material/public:",
            color="green",
        )
    else:
        st.badge(
            "Local sample search",
            icon=":material/science:",
            color="orange",
        )
        st.caption(
            "The local corpus is a small quantum-computing demo, not a general "
            "or authoritative source."
        )

    st.subheader("Workflow")
    st.mermaid_chart(
        """
        flowchart TD
            A["Planner<br/>Breaks the question into focused tasks"]
            B["Researcher<br/>Collects source-linked evidence"]
            C["Verifier<br/>Checks relevance and flags unsupported claims"]
            D["Synthesizer<br/>Builds a report and records limitations"]
            A --> B --> C --> D
        """,
        alt=(
            "Research flow: Planner creates tasks, Researcher collects "
            "evidence, Verifier checks support, and Synthesizer builds a report."
        ),
    )

st.title("Research desk", icon=":material/manage_search:")
st.write(
    "Move from a research question to a source-backed brief, with a clear "
    "record of verified findings and open gaps."
)

with st.container(border=True):
    st.subheader("Start a research brief")
    with st.form("research_request"):
        topic = st.text_area(
            "Research question",
            value=DEFAULT_TOPIC,
            placeholder="What would you like to investigate?",
            height=100,
            key="research_topic",
        )
        submitted = st.form_submit_button(
            "Generate brief",
            type="primary",
            icon=":material/arrow_forward:",
        )

if submitted:
    if not topic.strip():
        st.error("Enter a research question to continue.")
    else:
        settings = WorkflowSettings(
            online_llm=use_online_llm,
            web_search=use_live_search,
            openrouter_api_key=openrouter_api_key,
            serper_api_key=serper_api_key,
            memory_path=MEMORY_PATH,
        )
        try:
            from research_engine import run_integrated_workflow

            with st.spinner("Planning, researching, verifying, and synthesizing…"):
                result = run_integrated_workflow(
                    topic.strip(),
                    settings,
                    save_findings=save_findings,
                )
        except (OSError, ValueError) as error:
            st.error(f"The research run could not be completed: {error}")
        else:
            st.session_state["latest_run"] = {
                "topic": topic.strip(),
                "result": result,
                "save_findings": save_findings,
                "completed_at": datetime.now().astimezone().strftime(
                    "%b %d, %Y · %H:%M %Z"
                ),
            }

latest_run = st.session_state.get("latest_run")
if latest_run:
    result = latest_run["result"]
    verification = result["verification"]
    status = result["run_status"]

    st.subheader("Latest brief")
    st.caption(f"{latest_run['topic']} · Completed {latest_run['completed_at']}")

    metric_columns = st.columns(4)
    metric_columns[0].metric("Planned tasks", len(result["plan"]["tasks"]))
    metric_columns[1].metric(
        "Verified findings", len(verification["verified_findings"])
    )
    metric_columns[2].metric(
        "Open gaps", len(verification["missing_tasks"])
    )
    metric_columns[3].metric(
        "Memory entries", result["memory_saved_count"]
    )

    if status["llm_mode"] == "OFFLINE DEMO MODE":
        st.warning(
            status["llm_status_detail"],
            title="Deterministic planning and synthesis",
        )
    if "offline local sample" in status["source_modes"]:
        st.warning(
            "This run used the local quantum-computing sample documents. "
            "Treat them as illustrative excerpts, not published sources.",
            title="Sample sources used",
        )

    report_tab, evidence_tab, memory_tab, details_tab = st.tabs(
        ["Report", "Verified evidence", "Related memory", "Run details"]
    )
    with report_tab:
        st.download_button(
            "Download Markdown",
            data=result["report"],
            file_name=make_report_filename(latest_run["topic"]),
            mime="text/markdown",
            icon=":material/download:",
        )
        with st.container(border=True):
            st.markdown(result["report"])

    with evidence_tab:
        for finding in verification["verified_findings"]:
            with st.container(border=True):
                st.caption(finding["task"])
                st.write(finding["claim"])
                st.caption(
                    f"{finding['title']} · {finding['source']} · {finding['mode']}"
                )

        if verification["unsupported_claims"]:
            st.subheader("Unsupported claims")
            for claim in verification["unsupported_claims"]:
                st.warning(f"{claim['claim']} — {claim['reason']}")
        else:
            st.caption("No unsupported claims were included in this run.")

        if verification["missing_tasks"]:
            st.subheader("Tasks without matching evidence")
            for task in verification["missing_tasks"]:
                st.warning(task)

    with memory_tab:
        try:
            from research_engine import retrieve_memories

            related_memories = retrieve_memories(latest_run["topic"], MEMORY_PATH)
        except (OSError, ValueError) as error:
            st.error(f"Could not read saved research memory: {error}")
        else:
            if related_memories:
                st.caption(
                    f"Matching findings from `{MEMORY_PATH.name}`, including "
                    "the current run when saving was enabled."
                )
                for memory in related_memories:
                    with st.container(border=True):
                        st.write(memory["content"])
                        st.caption(
                            f"{memory.get('topic', '')} · "
                            f"{memory.get('source') or 'Source not recorded'}"
                        )
            else:
                st.caption("No saved findings match this question yet.")

    with details_tab:
        st.write("**Research question**")
        st.write(latest_run["topic"])
        st.write("**LLM mode**")
        st.write(status["llm_mode"])
        st.caption(status["llm_status_detail"])
        st.write("**Search sources**")
        st.write(", ".join(status["source_modes"]) or "No source results")
        st.write("**Agents**")
        st.write("Planner · Researcher · Verifier · Synthesizer")
        st.write("**Persistent memory**")
        st.write(
            f"{result['memory_saved_count']} verified findings indexed in "
            f"`{MEMORY_PATH.name}`."
            if latest_run["save_findings"]
            else "Saving was disabled for this run."
        )
else:
    st.space("medium")
    st.subheader("A clear path from question to evidence")
    columns = st.columns(4)
    stages = [
        ("01", "Plan", "Turn a broad question into focused lines of inquiry."),
        ("02", "Research", "Gather relevant findings and keep source labels."),
        ("03", "Verify", "Check task relevance and identify missing evidence."),
        ("04", "Synthesize", "Produce a report with transparent limitations."),
    ]
    for column, (number, title, description) in zip(columns, stages):
        with column.container(border=True, height="stretch"):
            st.caption(number)
            st.subheader(title)
            st.write(description)
