import streamlit as st
import re
import textstat

from humanizer.config import API_KEY, BASE_URL, AVAILABLE_MODELS, DEFAULT_MODEL
from humanizer.pipeline import HumanizationPipeline
from humanizer.text_analysis import get_text_analytics, compute_ai_risk_score

st.set_page_config(
    page_title="The Thesis Forge - AI Humanizer",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
        /* Hide default Streamlit header */
        header {visibility: hidden;}

        /* Custom Typography and Spacing */
        h1 {
            font-weight: 800;
            margin-bottom: 0.2rem;
            text-align: center;
        }
        .subtitle {
            color: #94a3b8;
            font-size: 1.1rem;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: 500;
        }

        /* Style text areas */
        .stTextArea textarea {
            background-color: #1a1f1b !important;
            color: #ffffff !important;
            border: 1px solid #2d8a4e !important;
            border-radius: 16px !important;
            padding: 1.5rem !important;
            font-size: 1.05rem !important;
            line-height: 1.6 !important;
            transition: all 0.3s ease-in-out;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        .stTextArea textarea:focus {
            border-color: #49C076 !important;
            box-shadow: 0 0 0 1px #49C076 !important;
            outline: none !important;
        }

        /* Buttons - Pill shaped, vibrant green */
        div.stButton > button:first-child {
            background-color: #49C076 !important;
            color: #000000 !important;
            border-radius: 24px !important;
            border: none !important;
            padding: 0.6rem 1.5rem !important;
            font-weight: 700 !important;
            transition: all 0.2s ease !important;
            width: 100% !important;
            box-shadow: 0 4px 10px rgba(73, 192, 118, 0.2) !important;
        }
        div.stButton > button:first-child:hover {
            background-color: #3BA863 !important;
            color: #ffffff !important;
            box-shadow: 0 6px 14px rgba(73, 192, 118, 0.4) !important;
            transform: translateY(-1px) !important;
        }

        /* Secondary Download Button */
        div.stDownloadButton > button:first-child {
            background-color: transparent !important;
            color: #49C076 !important;
            border: 1px solid #49C076 !important;
            border-radius: 24px !important;
            font-weight: 600 !important;
            width: 100% !important;
        }
        div.stDownloadButton > button:first-child:hover {
            background-color: rgba(73, 192, 118, 0.1) !important;
            color: #49C076 !important;
        }

        /* Metric Boxes for Analytics */
        div[data-testid="stMetric"] {
            background-color: #222924;
            padding: 1rem;
            border-radius: 12px;
            border: 1px solid #2d3730;
        }

        /* Progress stage indicator */
        .stage-indicator {
            padding: 0.5rem 1rem;
            border-radius: 8px;
            margin: 0.3rem 0;
            font-size: 0.9rem;
        }
        .stage-running {
            background-color: rgba(73, 192, 118, 0.15);
            border-left: 3px solid #49C076;
            color: #49C076;
        }
        .stage-complete {
            background-color: rgba(73, 192, 118, 0.05);
            border-left: 3px solid #2d8a4e;
            color: #94a3b8;
        }
        .stage-pending {
            background-color: rgba(255, 255, 255, 0.03);
            border-left: 3px solid #333;
            color: #555;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1>The Thesis Forge</h1>", unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Advanced multi-stage humanization pipeline '
    'that defeats AI detection systems.</div>',
    unsafe_allow_html=True
)

# Top Control Bar
st.write("")
col_space_left, col_controls, col_space_right = st.columns([1, 5, 1])

with col_controls:
    c1, c2, c3 = st.columns([1.5, 2, 1.5])
    with c1:
        model_choice = st.selectbox(
            "Model",
            AVAILABLE_MODELS,
            index=0
        )
    with c2:
        rewrite_intensity = st.slider(
            "Stealth Intensity",
            1, 5, 4,
            help="Level 1: Light touch (structural + lexical only). "
                 "Level 5: Maximum stealth (all stages at full power)."
        )
    with c3:
        st.write("")
        st.write("")
        trigger_rewrite = st.button("Humanize Text", type="primary")

# Advanced Settings
with st.expander("Advanced Settings - Pipeline Stage Toggles"):
    st.markdown(
        "Enable or disable individual pipeline stages. "
        "The intensity slider sets defaults, but you can override here."
    )
    from humanizer.config import INTENSITY_PROFILES
    profile = INTENSITY_PROFILES[rewrite_intensity]

    adv_c1, adv_c2, adv_c3, adv_c4, adv_c5 = st.columns(5)
    with adv_c1:
        enable_structural = st.checkbox(
            "Stage 1: Structural",
            value=profile["structural"],
            help="Sentence splitting, merging, clause reordering, paragraph restructuring."
        )
    with adv_c2:
        enable_lexical = st.checkbox(
            "Stage 2: Lexical",
            value=profile["lexical"],
            help="Replace predictable vocabulary with less common synonyms."
        )
    with adv_c3:
        enable_llm = st.checkbox(
            "Stage 3: LLM Rewrite",
            value=profile["llm_rewrite"],
            help="Multi-pass LLM rewriting for natural flow."
        )
    with adv_c4:
        enable_perplexity = st.checkbox(
            "Stage 4: Perplexity",
            value=profile["perplexity"],
            help="Vary sentence complexity to mimic human patterns."
        )
    with adv_c5:
        enable_postprocess = st.checkbox(
            "Stage 5: Postprocess",
            value=profile["postprocess"],
            help="Remove AI fingerprints, inject natural imperfections."
        )

st.markdown("<br>", unsafe_allow_html=True)

# Main Layout: Input | Output
col_left, col_right = st.columns(2, gap="large")

# LEFT COLUMN: INPUT
with col_left:
    st.markdown("### Original Draft")
    input_text = st.text_area(
        "Original Text",
        height=350,
        label_visibility="collapsed",
        placeholder="Paste your academic text here to humanize..."
    )

    if input_text:
        orig_stats = get_text_analytics(input_text)
        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Sentences", orig_stats['sentences'])
        m2.metric("Avg Length", f"{orig_stats['avg_len']} wds")
        m3.metric("Grade Level", orig_stats['grade'])
        m4.metric("Vocab Diversity", f"{orig_stats['vocabulary_diversity']:.2f}")
        m5.metric("AI Risk", f"{orig_stats['ai_risk']}%",
                  help="Internal estimate only - not a guarantee of detection results.")

# RIGHT COLUMN: OUTPUT
with col_right:
    st.markdown("### Humanized Output")

    if trigger_rewrite:
        if not input_text.strip():
            st.warning("Please paste some text on the left first.")
        else:
            # Build stage overrides from checkboxes
            stage_overrides = {
                "structural": enable_structural,
                "lexical": enable_lexical,
                "llm_rewrite": enable_llm,
                "perplexity": enable_perplexity,
                "postprocess": enable_postprocess,
            }

            # Progress display
            progress_placeholder = st.empty()
            stage_status = {}

            def update_progress(stage_name, status):
                stage_status[stage_name] = status
                progress_html = ""
                for name, stat in stage_status.items():
                    if stat == "running":
                        progress_html += (
                            f'<div class="stage-indicator stage-running">'
                            f'&#9654; {name}...</div>'
                        )
                    elif stat == "complete":
                        progress_html += (
                            f'<div class="stage-indicator stage-complete">'
                            f'&#10003; {name}</div>'
                        )
                progress_placeholder.markdown(progress_html, unsafe_allow_html=True)

            # Create pipeline
            pipeline = HumanizationPipeline(
                intensity=rewrite_intensity,
                model=model_choice,
                api_key=API_KEY,
                base_url=BASE_URL,
                stage_overrides=stage_overrides,
                progress_callback=update_progress,
            )

            stream_container = st.empty()
            final_text = ""

            try:
                if enable_llm:
                    # Use streaming mode for LLM stage
                    collected_chunks = []
                    postprocessed_text = None

                    for chunk in pipeline.process_stream(input_text):
                        if chunk == "\n\n__POSTPROCESSED__\n\n":
                            # Next chunk is the final post-processed text
                            postprocessed_text = ""
                            continue
                        if postprocessed_text is not None:
                            postprocessed_text = chunk
                            continue
                        collected_chunks.append(chunk)
                        stream_container.markdown(''.join(collected_chunks))

                    if postprocessed_text:
                        final_text = postprocessed_text
                        stream_container.markdown(final_text)
                    else:
                        final_text = ''.join(collected_chunks)
                else:
                    # No LLM stage - run synchronously
                    with st.spinner("Processing through pipeline stages..."):
                        final_text = pipeline.process(input_text)
                    stream_container.markdown(final_text)

                # Clear progress and show results
                progress_placeholder.empty()

                if final_text:
                    # Show post-processing analytics
                    st.markdown("<br>", unsafe_allow_html=True)
                    new_stats = get_text_analytics(final_text)
                    n1, n2, n3, n4, n5 = st.columns(5)
                    n1.metric("Sentences", new_stats['sentences'])
                    n2.metric("Avg Length", f"{new_stats['avg_len']} wds")

                    grade_delta = None
                    if (isinstance(new_stats['grade'], (int, float)) and
                            input_text and isinstance(orig_stats['grade'], (int, float))):
                        grade_delta = round(orig_stats['grade'] - new_stats['grade'], 1)
                    n3.metric("Grade Level", new_stats['grade'])

                    n4.metric("Vocab Diversity", f"{new_stats['vocabulary_diversity']:.2f}")

                    # AI risk comparison
                    risk_delta = None
                    if input_text:
                        risk_delta = orig_stats['ai_risk'] - new_stats['ai_risk']
                    n5.metric(
                        "AI Risk",
                        f"{new_stats['ai_risk']}%",
                        delta=f"-{risk_delta}%" if risk_delta and risk_delta > 0 else None,
                        delta_color="normal",
                        help="Internal estimate only - not a guarantee of detection results."
                    )

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.download_button(
                        label="Download Humanized Text",
                        data=final_text,
                        file_name="humanized_output.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.info(
            "Your humanized text will appear here. Adjust settings above "
            "and click **Humanize Text** to start the pipeline."
        )
