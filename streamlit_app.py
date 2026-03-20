"""
Streamlit demo UI for the AI Agent Evaluation Pipeline.

Run with:
    streamlit run streamlit_app.py

Make sure the FastAPI backend is running at http://localhost:8000
"""
import time
import uuid

import requests
import streamlit as st

# -- Config --------------------------------------------------------------------

st.set_page_config(
    page_title="AI Agent Evaluation Pipeline",
    layout="wide",
)

BASE_URL = "http://13.219.89.240:8000/api/v1"

# -- Helpers -------------------------------------------------------------------

def api_post(path: str, payload: dict) -> tuple[int, dict]:
    try:
        r = requests.post(f"{BASE_URL}{path}", json=payload, timeout=30)
        return r.status_code, r.json()
    except requests.exceptions.ConnectionError:
        return 0, {"detail": "Cannot connect to the API. Is the FastAPI server running?"}
    except Exception as e:
        return 0, {"detail": str(e)}


def api_get(path: str) -> tuple[int, dict | list]:
    try:
        r = requests.get(f"{BASE_URL}{path}", timeout=30)
        return r.status_code, r.json()
    except requests.exceptions.ConnectionError:
        return 0, {"detail": "Cannot connect to the API. Is the FastAPI server running?"}
    except Exception as e:
        return 0, {"detail": str(e)}


def api_patch(path: str, payload: dict) -> tuple[int, dict]:
    try:
        r = requests.patch(f"{BASE_URL}{path}", json=payload, timeout=10)
        return r.status_code, r.json()
    except requests.exceptions.ConnectionError:
        return 0, {"detail": "Cannot connect to the API. Is the FastAPI server running?"}
    except Exception as e:
        return 0, {"detail": str(e)}


def show_status(code: int, body: dict) -> None:
    if code == 0:
        st.error(body['detail'])
    elif code in (200, 202):
        st.success(f"Success ({code})")
    elif code == 404:
        st.warning("Not found (404)")
    else:
        st.error(f"Error {code}: {body.get('detail', body)}")


def score_label(score: float) -> str:
    if score >= 0.75:
        return "Good"
    if score >= 0.5:
        return "Fair"
    return "Poor"


SAMPLE_CONVERSATION_ID = "demo-conv-001"

# -- Sidebar -------------------------------------------------------------------

st.sidebar.title("Eval Pipeline")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Ingest Conversation",
        "Ingest Feedback",
        "Evaluate",
        "View Feedback",
        "Suggestions",
        "Meta-Evaluation",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption("FastAPI backend: `http://localhost:8000`")
st.sidebar.caption("[Swagger docs](http://localhost:8000/docs)")

# -----------------------------------------------------------------------------
# Page: Overview
# -----------------------------------------------------------------------------

if page == "Overview":
    st.title("AI Agent Evaluation Pipeline")
    st.markdown(
        "An end-to-end pipeline that automatically evaluates AI agent conversations, "
        "integrates human feedback, and generates improvement suggestions at scale."
    )

    st.markdown("### Pipeline Stages")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.info("**1. Ingest**\n\nConversation logs and human feedback are published to Kafka and persisted to MongoDB asynchronously.")
    with col2:
        st.info("**2. Evaluate**\n\n4 independent evaluators run concurrently: Heuristic, Tool Call, Coherence, and LLM Judge (Claude).")
    with col3:
        st.info("**3. Feedback Loop**\n\nHuman annotations are aggregated with inter-annotator kappa scoring and routed automatically.")
    with col4:
        st.info("**4. Meta-Evaluation**\n\nEvaluators are graded against human ground truth. Blind spots and calibration adjustments are computed.")

    st.markdown("### Try the Demo")
    st.markdown("""
1. **Ingest Conversation** — submit a sample agent conversation
2. **Ingest Feedback** — add a human rating and annotations for the same conversation
3. **Evaluate** — trigger evaluation and view scores from all 4 evaluators
4. **View Feedback** — see raw and aggregated feedback with kappa scores
5. **Suggestions** — browse improvement suggestions generated from evaluation
6. **Meta-Evaluation** — run the self-improvement loop across all data
""")

    st.markdown("### Architecture")
    st.code("""
HTTP Request → FastAPI Controller → Kafka (publish) → 202 Accepted
                                        ↓
                              Kafka Worker (async)
                                        ↓
                              Business Logic + MongoDB
    """, language="text")

# -----------------------------------------------------------------------------
# Page: Ingest Conversation
# -----------------------------------------------------------------------------

elif page == "Ingest Conversation":
    st.title("Ingest Conversation")
    st.markdown("Submit a multi-turn conversation log. It will be queued via Kafka and persisted to MongoDB.")

    with st.form("ingest_conversation_form"):
        st.markdown("#### Conversation Details")
        col1, col2 = st.columns(2)
        with col1:
            conversation_id = st.text_input("Conversation ID", value=SAMPLE_CONVERSATION_ID)
        with col2:
            agent_version = st.text_input("Agent Version", value="v1.0.0")

        col3, col4 = st.columns(2)
        with col3:
            total_latency = st.number_input("Total Latency (ms)", min_value=0, value=1200)
        with col4:
            mission_completed = st.checkbox("Mission Completed", value=True)

        st.markdown("#### Conversation Turns")
        st.caption("Add the turns of the conversation below.")

        turns = []

        with st.expander("Turn 1 — User", expanded=True):
            t1_content = st.text_area("Content", value="Can you search for the latest AI research papers on evaluation frameworks?", key="t1")
            turns.append({"turn_id": 1, "role": "user", "content": t1_content, "timestamp": "2026-03-20T10:00:00Z"})

        with st.expander("Turn 2 — Assistant (with tool call)", expanded=True):
            t2_content = st.text_area("Content", value="Sure! Let me search for that.", key="t2")
            include_tool = st.checkbox("Include tool call", value=True)
            tool_result = None
            if include_tool:
                tool_name = st.text_input("Tool Name", value="web_search")
                tool_param = st.text_input("Query Parameter", value="AI agent evaluation frameworks 2025")
                tool_result_text = st.text_input("Tool Result", value="Found 12 papers on LLM evaluation")
                tool_latency = st.number_input("Tool Latency (ms)", min_value=0, value=450)
                tool_result = {
                    "tool_name": tool_name,
                    "parameters": {"query": tool_param},
                    "result": {"summary": tool_result_text},
                    "latency_ms": tool_latency,
                }
            turn2 = {"turn_id": 2, "role": "assistant", "content": t2_content, "timestamp": "2026-03-20T10:00:01Z"}
            if tool_result:
                turn2["tool_calls"] = [tool_result]
            turns.append(turn2)

        with st.expander("Turn 3 — Assistant response", expanded=True):
            t3_content = st.text_area("Content", value="I found 12 recent papers. The most cited ones focus on LLM-as-judge evaluation, RAGAS frameworks, and multi-turn coherence benchmarks.", key="t3")
            turns.append({"turn_id": 3, "role": "assistant", "content": t3_content, "timestamp": "2026-03-20T10:00:02Z"})

        submitted = st.form_submit_button("Ingest Conversation", type="primary")

    if submitted:
        payload = {
            "conversation_id": conversation_id,
            "agent_version": agent_version,
            "turns": turns,
            "metadata": {
                "total_latency_ms": total_latency,
                "mission_completed": mission_completed,
            },
        }
        with st.spinner("Publishing to Kafka..."):
            code, body = api_post("/ingest/conversation", payload)
        show_status(code, body)
        if code == 202:
            st.json(body)
            st.info(f"Conversation ID `{conversation_id}` is queued. Use this ID in the other pages.")

# -----------------------------------------------------------------------------
# Page: Ingest Feedback
# -----------------------------------------------------------------------------

elif page == "Ingest Feedback":
    st.title("Ingest Feedback")
    st.markdown("Submit human feedback for a conversation. Annotations are aggregated with inter-annotator agreement scoring.")

    with st.form("ingest_feedback_form"):
        conversation_id = st.text_input("Conversation ID", value=SAMPLE_CONVERSATION_ID)

        st.markdown("#### User Rating")
        user_rating = st.slider("Rating (1 = poor, 5 = excellent)", 1, 5, 4)

        st.markdown("#### Ops Review")
        col1, col2 = st.columns(2)
        with col1:
            quality = st.selectbox("Quality", ["excellent", "good", "fair", "poor"], index=1)
        with col2:
            notes = st.text_input("Notes", value="Responses were helpful and factually accurate")

        st.markdown("#### Annotations")
        st.caption("Add typed labels from annotators. Multiple annotators for the same type will be used for kappa scoring.")

        annotations = []
        for i in range(1, 4):
            with st.expander(f"Annotation {i}", expanded=(i == 1)):
                col_a, col_b, col_c, col_d, col_e = st.columns([2, 2, 2, 2, 1])
                with col_a:
                    ann_type = st.text_input("Type", value="tone" if i == 1 else ("accuracy" if i == 2 else "tone"), key=f"ann_type_{i}")
                with col_b:
                    ann_label = st.text_input("Label", value="positive" if i <= 2 else "positive", key=f"ann_label_{i}")
                with col_c:
                    ann_annotator = st.text_input("Annotator ID", value=f"annotator-{i}", key=f"ann_ann_{i}")
                with col_d:
                    ann_confidence = st.slider("Confidence", 0.0, 1.0, 0.9 if i == 1 else (0.85 if i == 2 else 0.8), key=f"ann_conf_{i}")
                with col_e:
                    include_ann = st.checkbox("Include", value=True, key=f"ann_inc_{i}")

                if include_ann and ann_type and ann_label:
                    annotations.append({
                        "type": ann_type,
                        "label": ann_label,
                        "annotator_id": ann_annotator,
                        "confidence": ann_confidence,
                    })

        submitted = st.form_submit_button("Submit Feedback", type="primary")

    if submitted:
        payload = {
            "conversation_id": conversation_id,
            "user_rating": user_rating,
            "ops_review": {"quality": quality, "notes": notes},
            "annotations": annotations,
        }
        with st.spinner("Publishing to Kafka..."):
            code, body = api_post("/ingest/feedback", payload)
        show_status(code, body)
        if code == 202:
            st.json(body)
            st.info("Feedback is queued for aggregation. Check View Feedback in ~2 seconds once workers process it.")

# -----------------------------------------------------------------------------
# Page: Evaluate
# -----------------------------------------------------------------------------

elif page == "Evaluate":
    st.title("Evaluate")
    st.markdown("Trigger evaluation for a conversation and view scores from all four evaluators.")

    conversation_id = st.text_input("Conversation ID", value=SAMPLE_CONVERSATION_ID)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Trigger Evaluation", type="primary"):
            with st.spinner("Triggering evaluation via Kafka..."):
                code, body = api_post("/evaluations/trigger", {"conversation_id": conversation_id})
            show_status(code, body)
            if code == 202:
                st.info("Evaluation is running asynchronously. Click View Results in a few seconds.")

    with col2:
        if st.button("View Results"):
            with st.spinner("Fetching results..."):
                code, body = api_get(f"/evaluations/conversation/{conversation_id}")

            if code == 0:
                show_status(code, body)
            elif code == 200 and isinstance(body, list) and len(body) > 0:
                result = body[-1]  # most recent evaluation
                scores = result.get("scores", {})

                st.success(f"Showing latest evaluation — `{result['evaluation_id']}`")

                st.markdown("### Scores")
                m1, m2, m3, m4, m5 = st.columns(5)
                overall = scores.get("overall", 0)
                m1.metric("Overall", f"{overall:.2f}", delta=None)
                m1.caption(score_label(overall))

                for col, key, label in [
                    (m2, "response_quality", "LLM Judge"),
                    (m3, "tool_accuracy",    "Tool Accuracy"),
                    (m4, "coherence",        "Coherence"),
                    (m5, "heuristic",        "Heuristic"),
                ]:
                    val = scores.get(key)
                    if val is not None:
                        col.metric(label, f"{val:.2f}")
                        col.caption(score_label(val))
                    else:
                        col.metric(label, "N/A")

                # Bar chart
                score_keys = ["response_quality", "tool_accuracy", "coherence", "heuristic"]
                chart_data = {
                    k.replace("_", " ").title(): scores.get(k, 0)
                    for k in score_keys
                    if scores.get(k) is not None
                }
                st.bar_chart(chart_data)

                # Tool details
                tool_details = scores.get("tool_details")
                if tool_details:
                    st.markdown("### Tool Evaluation Details")
                    tc1, tc2, tc3, tc4 = st.columns(4)
                    tc1.metric("Selection Accuracy", f"{tool_details['selection_accuracy']:.2f}")
                    tc2.metric("Parameter Accuracy", f"{tool_details['parameter_accuracy']:.2f}")
                    tc3.metric("Execution Success", f"{tool_details['execution_success']:.2f}")
                    tc4.metric("Hallucinated Params", tool_details["hallucinated_params"])

                # Coherence details
                coh_details = scores.get("coherence_details")
                if coh_details:
                    st.markdown("### Coherence Details")
                    cc1, cc2, cc3 = st.columns(3)
                    cc1.metric("Consistency Score", f"{coh_details['consistency_score']:.2f}")
                    cc2.metric("Context Retention", f"{coh_details['context_retention']:.2f}")
                    cc3.metric("Contradiction Count", coh_details["contradiction_count"])

                # Issues
                issues = result.get("issues_detected", [])
                if issues:
                    st.markdown("### Issues Detected")
                    severity_label = {"info": "[info]", "warning": "[warning]", "critical": "[critical]"}
                    for issue in issues:
                        label = severity_label.get(issue["severity"], "")
                        st.markdown(f"- **{issue['type']}** `{issue['severity']}`: {issue.get('description', issue.get('message', ''))}")
                else:
                    st.success("No issues detected")

            elif code == 200 and isinstance(body, list) and len(body) == 0:
                st.warning("No evaluation results yet. Trigger an evaluation first and wait a moment for the worker to process it.")
            else:
                show_status(code, body if isinstance(body, dict) else {"detail": str(body)})

# -----------------------------------------------------------------------------
# Page: View Feedback
# -----------------------------------------------------------------------------

elif page == "View Feedback":
    st.title("View Feedback")
    st.markdown("View raw and aggregated feedback for a conversation.")

    conversation_id = st.text_input("Conversation ID", value=SAMPLE_CONVERSATION_ID)

    tab_raw, tab_agg = st.tabs(["Raw Feedback", "Aggregated (Kappa + Routing)"])

    with tab_raw:
        if st.button("Fetch Raw Feedback"):
            code, body = api_get(f"/feedback/{conversation_id}")
            if code == 200:
                st.markdown("#### User Rating")
                rating = body.get("user_rating")
                if rating:
                    st.markdown(f"**{rating} / 5**")

                ops = body.get("ops_review")
                if ops:
                    st.markdown(f"**Ops Review:** `{ops.get('quality', 'N/A')}` — {ops.get('notes', '')}")

                annotations = body.get("annotations") or []
                if annotations:
                    st.markdown(f"#### Annotations ({len(annotations)})")
                    st.dataframe(annotations, use_container_width=True)
            else:
                show_status(code, body)

    with tab_agg:
        if st.button("Fetch Aggregated Feedback"):
            code, body = api_get(f"/feedback/{conversation_id}/aggregated")
            if code == 200:
                col1, col2, col3 = st.columns(3)
                kappa = body.get("overall_kappa", 0)
                routing = body.get("routing", "unknown")
                col1.metric("Overall Kappa", f"{kappa:.3f}")
                col2.metric("Routing Decision", routing)
                col3.metric("User Rating", body.get("user_rating", "N/A"))

                routing_note = "High agreement: auto-labelled" if routing == "auto_label" else "Low agreement: sent to human review"
                st.markdown(f"**{routing.replace('_', ' ').title()}** — {routing_note}")

                consensus = body.get("annotation_consensus", [])
                if consensus:
                    st.markdown("#### Annotation Consensus")
                    for item in consensus:
                        with st.expander(f"Type: `{item['type']}` — {item['consensus_label']}"):
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Kappa", f"{item['kappa']:.3f}")
                            c2.metric("Agreement", f"{item['agreement_score']:.1%}")
                            c3.metric("Confidence", f"{item['weighted_confidence']:.2f}")
                            c4.metric("Annotators", item["annotator_count"])
            elif code == 404:
                st.warning("Aggregated feedback not found. Either no feedback has been submitted, or the worker hasn't processed it yet.")
            else:
                show_status(code, body)

# -----------------------------------------------------------------------------
# Page: Suggestions
# -----------------------------------------------------------------------------

elif page == "Suggestions":
    st.title("Improvement Suggestions")
    st.markdown("Browse suggestions generated from evaluation results. Update their status as you review them.")

    col1, col2, col3 = st.columns(3)
    with col1:
        filter_status = st.selectbox("Filter by Status", ["(all)", "pending", "applied", "rejected", "expired"])
    with col2:
        filter_type = st.selectbox("Filter by Type", ["(all)", "prompt", "tool"])
    with col3:
        filter_version = st.text_input("Filter by Agent Version", placeholder="e.g. v1.0.0")

    params = []
    if filter_status != "(all)":
        params.append(f"status={filter_status}")
    if filter_type != "(all)":
        params.append(f"type={filter_type}")
    if filter_version:
        params.append(f"agent_version={filter_version}")
    query = "?" + "&".join(params) if params else ""

    if st.button("Load Suggestions"):
        code, body = api_get(f"/suggestions{query}")
        if code == 200 and isinstance(body, list):
            if not body:
                st.info("No suggestions found. Run an evaluation first.")
            else:
                st.success(f"Found {len(body)} suggestion(s)")
                for s in body:
                    with st.expander(f"[{s['type']}] {s['suggestion'][:80]}... — {s['status']}"):
                        st.markdown(f"**ID:** `{s['suggestion_id']}`")
                        st.markdown(f"**Type:** `{s['type']}`  |  **Agent Version:** `{s.get('agent_version', 'N/A')}`")
                        st.markdown(f"**Confidence:** {s['confidence']:.0%}  |  **Expected Impact:** {s.get('expected_impact', 'N/A')}")
                        st.markdown(f"**Rationale:** {s['rationale']}")
                        conv_ids = s.get("conversation_ids", [])
                        if conv_ids:
                            st.markdown(f"**Linked conversations:** {', '.join(f'`{c}`' for c in conv_ids)}")

                        new_status = st.selectbox(
                            "Update Status",
                            ["pending", "applied", "rejected", "expired"],
                            index=["pending", "applied", "rejected", "expired"].index(s["status"]),
                            key=f"status_{s['suggestion_id']}",
                        )
                        if st.button("Save Status", key=f"save_{s['suggestion_id']}"):
                            code2, body2 = api_patch(
                                f"/suggestions/{s['suggestion_id']}/status",
                                {"status": new_status},
                            )
                            if code2 == 200:
                                st.success(f"Updated to `{new_status}`")
                            else:
                                show_status(code2, body2)
        else:
            show_status(code, body if isinstance(body, dict) else {"detail": str(body)})

# -----------------------------------------------------------------------------
# Page: Meta-Evaluation
# -----------------------------------------------------------------------------

elif page == "Meta-Evaluation":
    st.title("Meta-Evaluation")
    st.markdown(
        "Compare automated evaluator scores against human ground truth. "
        "Computes precision, recall, F1, and Pearson correlation per evaluator, "
        "detects blind spots, and generates calibration adjustments."
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Run Meta-Evaluation", type="primary"):
            with st.spinner("Running meta-evaluation across all paired conversations..."):
                code, body = api_post("/meta-evaluations/run", {})
            if code == 200:
                st.success(f"Meta-evaluation complete — `{body['meta_evaluation_id']}`")
                st.session_state["meta_result"] = body
            else:
                show_status(code, body)

    with col2:
        if st.button("Load Latest"):
            code, body = api_get("/meta-evaluations/latest")
            if code == 200:
                st.success(f"Loaded — `{body['meta_evaluation_id']}`")
                st.session_state["meta_result"] = body
            else:
                show_status(code, body)

    result = st.session_state.get("meta_result")
    if result:
        st.markdown(f"**Sample size:** {result['sample_size']} paired conversations")

        if result["sample_size"] == 0:
            st.warning("No paired data yet. Ingest conversations + feedback and run an evaluation first, then come back here.")
        else:
            # Evaluator metrics table
            metrics = result.get("evaluator_metrics", [])
            if metrics:
                st.markdown("### Evaluator Performance vs Human Ground Truth")
                st.caption("Positive class = 'bad' conversation (score < 0.5). F1 measures how well each evaluator identifies bad conversations that humans also rated poorly.")

                all_f1_zero = all(m["f1_score"] == 0.0 for m in metrics)
                if all_f1_zero:
                    st.info(
                        "**F1 = 0 for all evaluators.** This is expected when all conversations in the sample "
                        "score above 0.5 (the 'bad' threshold) for both evaluators and humans — there are no negative "
                        "examples to detect. To see non-zero F1, ingest a few conversations with low quality and "
                        "submit feedback with a low rating (1-2 stars). "
                        "**Pearson correlation** (below) is the more useful signal with small or all-positive samples."
                    )

                cols = st.columns(len(metrics))
                for col, m in zip(cols, metrics):
                    col.markdown(f"#### {m['evaluator_type'].replace('_', ' ').title()}")
                    col.metric("Correlation r", f"{m['correlation_with_human']:.2f}")
                    col.metric("F1", f"{m['f1_score']:.2f}")
                    col.metric("Precision", f"{m['precision']:.2f}")
                    col.metric("Recall", f"{m['recall']:.2f}")
                    col.caption(f"n={m['sample_size']}")

                st.markdown("#### Pearson Correlation with Human Scores")
                corr_data = {m["evaluator_type"].replace("_", " ").title(): m["correlation_with_human"] for m in metrics}
                st.bar_chart(corr_data)

                st.markdown("#### F1 Score by Evaluator")
                chart_data = {m["evaluator_type"].replace("_", " ").title(): m["f1_score"] for m in metrics}
                st.bar_chart(chart_data)

            # Blind spots
            blind_spots = result.get("blind_spots", [])
            if blind_spots:
                st.markdown("### Blind Spots")
                st.caption("Annotation types that appear in conversations the evaluators rated as good but humans rated as bad.")
                bs_data = [
                    {
                        "Annotation Type": b["annotation_type"],
                        "Miss Rate": f"{b['miss_rate']:.1%}",
                        "Example Count": b["example_count"],
                    }
                    for b in blind_spots
                ]
                st.dataframe(bs_data, use_container_width=True)
            else:
                st.info("No blind spots detected.")

            # Calibration adjustments
            adjustments = result.get("calibration_adjustments", [])
            if adjustments:
                st.markdown("### Calibration Adjustments")
                st.caption("Recommended weight multipliers based on each evaluator's F1 score.")
                for adj in adjustments:
                    mult = adj["weight_multiplier"]
                    direction = "increase" if mult > 1.0 else ("decrease" if mult < 1.0 else "no change")
                    st.markdown(
                        f"- **{adj['evaluator_type'].replace('_', ' ').title()}** — "
                        f"multiply weight by **{mult}x** ({direction}) — {adj['reason']}"
                    )
