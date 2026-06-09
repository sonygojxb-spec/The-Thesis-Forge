import streamlit as st
import re

from humanizer.config import (
    API_KEY, BASE_URL, AVAILABLE_MODELS, DEFAULT_MODEL,
    INTENSITY_PROFILES, ACADEMIC_ROLES, ACADEMIC_FIELDS,
    STYLE_PREFERENCES, CRITIC_DEFAULT_THRESHOLD, CRITIC_MAX_RETRIES,
)
from humanizer.pipeline import HumanizationPipeline
from humanizer.text_analysis import get_text_analytics, compute_ai_risk_score
from humanizer.voice_analysis import get_voice_profile
from humanizer.critic import CriticLoop
from humanizer.identity import AcademicIdentity
from humanizer.cowrite import CoWriter
from humanizer.classifier import detection_risk_score
from humanizer.config_serializer import ConfigSerializer, PipelineConfig, ConfigError

# --- Page Config ---
st.set_page_config(
    page_title="The Thesis Forge - AI Humanizer",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Session State Initialization ---
if "identity" not in st.session_state:
    st.session_state.identity = None
if "style_preferences" not in st.session_state:
    st.session_state.style_preferences = None
if "critic_history" not in st.session_state:
    st.session_state.critic_history = []
if "humanized_output" not in st.session_state:
    st.session_state.humanized_output = ""
if "cowrite_output" not in st.session_state:
    st.session_state.cowrite_output = ""
if "final_similarity" not in st.session_state:
    st.session_state.final_similarity = None
if "original_input_text" not in st.session_state:
    st.session_state.original_input_text = ""

# --- Custom CSS ---
st.markdown("""
    <style>
        header {visibility: hidden;}

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

        div[data-testid="stMetric"] {
            background-color: #222924;
            padding: 1rem;
            border-radius: 12px;
            border: 1px solid #2d3730;
        }

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

        .voice-metric {
            text-align: center;
            padding: 0.5rem;
        }
        .voice-metric-value {
            font-size: 1.4rem;
            font-weight: 700;
            color: #49C076;
        }
        .voice-metric-label {
            font-size: 0.75rem;
            color: #94a3b8;
        }
        .attempt-box {
            padding: 0.5rem 1rem;
            border-radius: 8px;
            margin: 0.3rem 0;
            font-size: 0.9rem;
            border-left: 3px solid #2d8a4e;
            background-color: rgba(73, 192, 118, 0.05);
        }
        .attempt-success {
            border-left: 3px solid #49C076;
            background-color: rgba(73, 192, 118, 0.15);
        }
        .attempt-fail {
            border-left: 3px solid #f59e0b;
            background-color: rgba(245, 158, 11, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("<h1>The Thesis Forge</h1>", unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Advanced multi-stage humanization pipeline '
    'that defeats AI detection systems.</div>',
    unsafe_allow_html=True
)


# --- Identity / Role Conditioning Section ---
with st.expander("Academic Identity (applies to all modes)", expanded=False):
    id_c1, id_c2, id_c3, id_c4 = st.columns(4)
    with id_c1:
        id_role = st.selectbox("Role", ["None"] + ACADEMIC_ROLES, index=0)
    with id_c2:
        id_field = st.selectbox("Field", ["None"] + ACADEMIC_FIELDS, index=0)
    with id_c3:
        id_institution = st.text_input("Institution (optional)", value="")
    with id_c4:
        id_style = st.selectbox("Style Preference", STYLE_PREFERENCES, index=0)

    if id_role != "None" and id_field != "None":
        st.session_state.identity = AcademicIdentity(
            role=id_role,
            field=id_field,
            institution=id_institution if id_institution.strip() else None,
            style_preference=id_style,
        )
        st.caption(f"Identity active: {id_role} in {id_field} ({id_style})")
    else:
        st.session_state.identity = None
        st.caption("No identity set. Select both Role and Field to enable.")


# --- Helper: Style preferences injection ---
def get_style_instructions():
    """Build style preference instructions for LLM prompt injection."""
    prefs = st.session_state.style_preferences
    if not prefs:
        return ""
    lines = []
    if prefs.get("avg_sentence_length_preference"):
        lines.append(f"Preferred average sentence length: {prefs['avg_sentence_length_preference']}")
    if prefs.get("hedge_preference") and prefs["hedge_preference"] != "same":
        lines.append(f"Use {prefs['hedge_preference']} hedging expressions")
    if prefs.get("formality_direction"):
        lines.append(f"Adjust formality: {prefs['formality_direction']}")
    if prefs.get("vocabulary_notes"):
        lines.append(f"Vocabulary notes: {prefs['vocabulary_notes']}")
    if not lines:
        return ""
    return "STYLE PREFERENCES FROM USER FEEDBACK:\n" + "\n".join(lines)


# --- Helper: Detect style preferences from edits ---
def detect_style_preferences(original, edited):
    """Compare original and edited text to detect style preference patterns."""
    prefs = {}

    orig_sentences = [s.strip() for s in re.split(r'[.!?]+', original) if s.strip()]
    edit_sentences = [s.strip() for s in re.split(r'[.!?]+', edited) if s.strip()]

    # Average sentence length preference
    orig_avg = sum(len(s.split()) for s in orig_sentences) / max(len(orig_sentences), 1)
    edit_avg = sum(len(s.split()) for s in edit_sentences) / max(len(edit_sentences), 1)

    if edit_avg < orig_avg * 0.85:
        prefs["avg_sentence_length_preference"] = "shorter sentences"
    elif edit_avg > orig_avg * 1.15:
        prefs["avg_sentence_length_preference"] = "longer sentences"
    else:
        prefs["avg_sentence_length_preference"] = ""

    # Hedge preference
    hedge_words = ["might", "perhaps", "seems", "could", "arguably", "possibly", "may", "likely"]
    orig_hedges = sum(1 for w in original.lower().split() if w in hedge_words)
    edit_hedges = sum(1 for w in edited.lower().split() if w in hedge_words)

    if edit_hedges > orig_hedges + 2:
        prefs["hedge_preference"] = "more"
    elif edit_hedges < orig_hedges - 2:
        prefs["hedge_preference"] = "less"
    else:
        prefs["hedge_preference"] = "same"

    # Formality direction
    informal_markers = ["basically", "kind of", "sort of", "pretty much", "stuff", "things"]
    orig_informal = sum(1 for m in informal_markers if m in original.lower())
    edit_informal = sum(1 for m in informal_markers if m in edited.lower())

    if edit_informal > orig_informal:
        prefs["formality_direction"] = "less formal"
    elif edit_informal < orig_informal:
        prefs["formality_direction"] = "more formal"
    else:
        prefs["formality_direction"] = ""

    # Vocabulary notes (detect if user consistently changes specific words)
    orig_words = set(original.lower().split())
    edit_words = set(edited.lower().split())
    # Filter out common function words (stopwords) that dominate set differences
    # during structural edits and do not represent vocabulary preferences
    function_words = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "shall",
        "should", "may", "might", "can", "could", "must", "ought",
        "i", "me", "my", "mine", "we", "us", "our", "ours",
        "you", "your", "yours", "he", "him", "his", "she", "her", "hers",
        "it", "its", "they", "them", "their", "theirs",
        "this", "that", "these", "those", "who", "whom", "which", "what",
        "and", "but", "or", "nor", "for", "yet", "so",
        "in", "on", "at", "to", "from", "by", "with", "of", "about",
        "into", "through", "during", "before", "after", "above", "below",
        "between", "under", "over", "up", "down", "out", "off", "then",
        "than", "if", "as", "not", "no", "all", "each", "every",
        "both", "few", "more", "most", "other", "some", "such",
        "only", "own", "same", "too", "very", "just",
    }
    new_words = edit_words - orig_words - function_words
    removed_words = orig_words - edit_words - function_words

    notes = []
    if len(new_words) > 5:
        sample = list(new_words)[:5]
        notes.append(f"User added words like: {', '.join(sample)}")
    if len(removed_words) > 5:
        sample = list(removed_words)[:5]
        notes.append(f"User removed words like: {', '.join(sample)}")
    prefs["vocabulary_notes"] = "; ".join(notes) if notes else ""

    return prefs


# --- Helper: Display voice profile comparison ---
def display_voice_comparison(input_profile, output_profile):
    """Display before/after voice profile comparison."""
    st.markdown("#### Voice Profile Comparison")

    col_labels, col_before, col_after = st.columns([2, 1, 1])
    with col_labels:
        st.markdown("**Metric**")
    with col_before:
        st.markdown("**Before**")
    with col_after:
        st.markdown("**After**")

    metrics = [
        ("Hedges", "hedges"),
        ("Boosters", "boosters"),
        ("Pronouns", "pronouns"),
        ("Engagement Markers", "engagement"),
        ("Type-Token Ratio", "ttr"),
    ]

    for label, key in metrics:
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            st.text(label)
        with c2:
            val = input_profile[key]
            st.text(f"{val:.3f}" if isinstance(val, float) else str(val))
        with c3:
            val = output_profile[key]
            st.text(f"{val:.3f}" if isinstance(val, float) else str(val))

    # Idiosyncrasies
    st.markdown("**Idiosyncrasies:**")
    idio_keys = ["parentheticals", "em_dashes", "fragments", "questions"]
    ic1, ic2, ic3 = st.columns([2, 1, 1])
    with ic1:
        for k in idio_keys:
            st.text(k.replace("_", " ").title())
    with ic2:
        for k in idio_keys:
            st.text(str(input_profile["idiosyncrasies"][k]))
    with ic3:
        for k in idio_keys:
            st.text(str(output_profile["idiosyncrasies"][k]))


# --- Helper: Display revision feedback section ---
def display_revision_feedback(output_text, key_prefix):
    """Show editable output and save preferences button."""
    st.markdown("---")
    st.markdown("#### Revision Feedback")
    st.caption("Edit the output below to refine your preferences. Click Save to track your style.")

    edited_text = st.text_area(
        "Edit Output",
        value=output_text,
        height=250,
        key=f"{key_prefix}_revision_area",
        label_visibility="collapsed",
    )

    if st.button("Save Style Preferences", key=f"{key_prefix}_save_prefs"):
        if edited_text.strip() != output_text.strip():
            prefs = detect_style_preferences(output_text, edited_text)
            st.session_state.style_preferences = prefs
            st.success("Style preferences saved! They will be applied on subsequent runs.")
            st.json(prefs)
        else:
            st.info("No changes detected. Edit the text above before saving.")

    if st.session_state.style_preferences:
        with st.expander("Current Style Preferences", expanded=False):
            st.json(st.session_state.style_preferences)


# --- Main Tabs ---
tab_humanize, tab_cowrite = st.tabs(["Humanize", "Co-Write"])


# =============================================================================
# HUMANIZE TAB
# =============================================================================
with tab_humanize:
    # Top Control Bar
    st.write("")
    col_space_left, col_controls, col_space_right = st.columns([1, 5, 1])

    with col_controls:
        c1, c2, c3 = st.columns([1.5, 2, 1.5])
        with c1:
            model_choice = st.selectbox(
                "Model",
                AVAILABLE_MODELS,
                index=0,
                key="humanize_model"
            )
        with c2:
            rewrite_intensity = st.slider(
                "Stealth Intensity",
                1, 5, 4,
                help="Level 1: Light touch (structural + lexical only). "
                     "Level 5: Maximum stealth (all stages at full power).",
                key="humanize_intensity"
            )
        with c3:
            st.write("")
            st.write("")
            trigger_rewrite = st.button("Humanize Text", type="primary", key="humanize_btn")

    # Advanced Settings
    with st.expander("Advanced Settings - Pipeline Stage Toggles"):
        st.markdown(
            "Enable or disable individual pipeline stages. "
            "The intensity slider sets defaults, but you can override here."
        )
        profile = INTENSITY_PROFILES[rewrite_intensity]

        # --- Existing five stages ---
        st.markdown("**Original Stages**")
        adv_c1, adv_c2, adv_c3, adv_c4, adv_c5 = st.columns(5)
        with adv_c1:
            enable_structural = st.checkbox(
                "Stage 1: Structural",
                value=profile["structural"],
                help="Sentence splitting, merging, clause reordering, paragraph restructuring.",
                key="stage_structural"
            )
        with adv_c2:
            enable_lexical = st.checkbox(
                "Stage 2: Lexical",
                value=profile["lexical"],
                help="Replace predictable vocabulary with less common synonyms.",
                key="stage_lexical"
            )
        with adv_c3:
            enable_llm = st.checkbox(
                "Stage 3: LLM Rewrite",
                value=profile["llm_rewrite"],
                help="Multi-pass LLM rewriting for natural flow.",
                key="stage_llm"
            )
        with adv_c4:
            enable_perplexity = st.checkbox(
                "Stage 4: Perplexity",
                value=profile["perplexity"],
                help="Vary sentence complexity to mimic human patterns.",
                key="stage_perplexity"
            )
        with adv_c5:
            enable_postprocess = st.checkbox(
                "Stage 5: Postprocess",
                value=profile["postprocess"],
                help="Remove AI fingerprints, inject natural imperfections.",
                key="stage_postprocess"
            )

        # --- Nine new capability stages ---
        st.markdown("**Advanced Capabilities**")
        new_c1, new_c2, new_c3 = st.columns(3)
        with new_c1:
            enable_semantic_transform = st.checkbox(
                "Semantic Transformation",
                value=profile.get("semantic_transform_enabled", False),
                help="Surface-form transforms with strict 0.90 similarity gate.",
                key="stage_semantic_transform"
            )
            enable_iterative_paraphrase = st.checkbox(
                "Iterative Paraphrasing",
                value=profile.get("iterative_paraphrase_enabled", False),
                help="Multi-pass LLM paraphrasing for progressive divergence.",
                key="stage_iterative_paraphrase"
            )
            enable_retrieval_augmented = st.checkbox(
                "Retrieval-Augmented Rewriting",
                value=profile.get("retrieval_augmented_enabled", False),
                help="Ground rewrites in human-written reference passages.",
                key="stage_retrieval_augmented"
            )
        with new_c2:
            enable_stylometric = st.checkbox(
                "Stylometric Obfuscation",
                value=profile.get("stylometric_enabled", False),
                help="Disrupt stylometric fingerprints (sentence length, function words, TTR).",
                key="stage_stylometric"
            )
            enable_perplexity_optimize = st.checkbox(
                "Perplexity Optimization",
                value=profile.get("perplexity_optimize_enabled", False),
                help="Tune text toward a target perplexity and burstiness profile.",
                key="stage_perplexity_optimize"
            )
            enable_adversarial = st.checkbox(
                "Adversarial Rewriting",
                value=profile.get("adversarial_enabled", False),
                help="Rewrite text to specifically evade detector signals.",
                key="stage_adversarial"
            )
        with new_c3:
            enable_error_injection = st.checkbox(
                "Error Injection",
                value=profile.get("error_injection_enabled", False),
                help="Inject controlled human-like imperfections (punctuation, whitespace).",
                key="stage_error_injection"
            )
            enable_detector_optimize = st.checkbox(
                "Detector Optimization",
                value=profile.get("detector_optimize_enabled", False),
                help="Closed-loop optimization to minimize detection probability.",
                key="stage_detector_optimize"
            )
            enable_classifier = st.checkbox(
                "AI Classifier",
                value=profile.get("classifier_enabled", False),
                help="Use transformer-based classifier for risk scoring (enables detector stages).",
                key="stage_classifier"
            )

        # --- Target Perplexity Profile ---
        st.markdown("**Target Perplexity Profile**")
        ppx_c1, ppx_c2 = st.columns(2)
        with ppx_c1:
            target_perplexity_mean = st.number_input(
                "Target Perplexity Mean",
                min_value=1.0,
                value=float(profile.get("target_perplexity_mean", 60.0)),
                step=5.0,
                help="Target mean perplexity for human-like text (typical: 50-80).",
                key="target_perplexity_mean"
            )
        with ppx_c2:
            target_perplexity_variance = st.number_input(
                "Target Perplexity Variance",
                min_value=0.0,
                value=float(profile.get("target_perplexity_variance", 15.0)),
                step=1.0,
                help="Target cross-sentence perplexity variance (typical: 10-25).",
                key="target_perplexity_variance"
            )

    # Configuration Save/Load
    with st.expander("Configuration Save/Load"):
        cfg_c1, cfg_c2 = st.columns(2)

        with cfg_c1:
            st.markdown("**Save Configuration**")
            # Build PipelineConfig from the current UI state
            _current_profile = INTENSITY_PROFILES[rewrite_intensity]
            _current_config = PipelineConfig(
                intensity=rewrite_intensity,
                semantic_transform_enabled=enable_semantic_transform,
                iterative_paraphrase_enabled=enable_iterative_paraphrase,
                retrieval_augmented_enabled=enable_retrieval_augmented,
                stylometric_enabled=enable_stylometric,
                perplexity_optimize_enabled=enable_perplexity_optimize,
                adversarial_enabled=enable_adversarial,
                error_injection_enabled=enable_error_injection,
                detector_optimize_enabled=enable_detector_optimize,
                classifier_enabled=enable_classifier,
                semantic_transform_aggression=float(_current_profile.get("semantic_transform_aggression", 0.0)),
                iterative_paraphrase_aggression=float(_current_profile.get("iterative_paraphrase_aggression", 0.0)),
                retrieval_augmented_aggression=float(_current_profile.get("retrieval_augmented_aggression", 0.0)),
                stylometric_aggression=float(_current_profile.get("stylometric_aggression", 0.0)),
                perplexity_optimize_aggression=float(_current_profile.get("perplexity_optimize_aggression", 0.0)),
                adversarial_aggression=float(_current_profile.get("adversarial_aggression", 0.0)),
                error_injection_aggression=float(_current_profile.get("error_injection_aggression", 0.0)),
                detector_optimize_aggression=float(_current_profile.get("detector_optimize_aggression", 0.0)),
                target_perplexity_mean=target_perplexity_mean,
                target_perplexity_variance=target_perplexity_variance,
            )
            _config_json = ConfigSerializer.serialize(_current_config)
            st.download_button(
                label="Save Config",
                data=_config_json,
                file_name="pipeline_config.json",
                mime="application/json",
                use_container_width=True,
                key="config_save_btn"
            )

        with cfg_c2:
            st.markdown("**Load Configuration**")
            uploaded_config = st.file_uploader(
                "Upload config JSON",
                type=["json"],
                key="config_upload",
                label_visibility="collapsed",
            )
            if uploaded_config is not None:
                try:
                    config_blob = uploaded_config.read().decode("utf-8")
                    loaded_config = ConfigSerializer.deserialize(config_blob)
                    # Update session state with loaded values
                    st.session_state["humanize_intensity"] = loaded_config.intensity
                    st.session_state["stage_semantic_transform"] = loaded_config.semantic_transform_enabled
                    st.session_state["stage_iterative_paraphrase"] = loaded_config.iterative_paraphrase_enabled
                    st.session_state["stage_retrieval_augmented"] = loaded_config.retrieval_augmented_enabled
                    st.session_state["stage_stylometric"] = loaded_config.stylometric_enabled
                    st.session_state["stage_perplexity_optimize"] = loaded_config.perplexity_optimize_enabled
                    st.session_state["stage_adversarial"] = loaded_config.adversarial_enabled
                    st.session_state["stage_error_injection"] = loaded_config.error_injection_enabled
                    st.session_state["stage_detector_optimize"] = loaded_config.detector_optimize_enabled
                    st.session_state["stage_classifier"] = loaded_config.classifier_enabled
                    st.session_state["target_perplexity_mean"] = loaded_config.target_perplexity_mean
                    st.session_state["target_perplexity_variance"] = loaded_config.target_perplexity_variance
                    st.success("Configuration loaded successfully!")
                    st.rerun()
                except ConfigError as e:
                    st.error(f"Invalid field: {e.field}")

    # Critic Loop Settings
    with st.expander("Critic Loop Settings"):
        critic_c1, critic_c2 = st.columns(2)
        with critic_c1:
            enable_critic = st.checkbox(
                "Enable Auto-Critic",
                value=False,
                help="Automatically re-run the pipeline if AI risk score is above threshold.",
                key="enable_critic"
            )
        with critic_c2:
            critic_threshold = st.slider(
                "Risk Threshold (%)",
                10, 80, CRITIC_DEFAULT_THRESHOLD,
                help="Re-run pipeline if AI risk score exceeds this value.",
                key="critic_threshold"
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
            placeholder="Paste your academic text here to humanize...",
            key="humanize_input"
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
            if not input_text or not input_text.strip():
                st.warning("Please paste some text on the left first.")
            else:
                # Build stage overrides from checkboxes
                stage_overrides = {
                    "structural": enable_structural,
                    "lexical": enable_lexical,
                    "llm_rewrite": enable_llm,
                    "perplexity": enable_perplexity,
                    "postprocess": enable_postprocess,
                    "semantic_transform": enable_semantic_transform,
                    "iterative_paraphrase": enable_iterative_paraphrase,
                    "retrieval_augmented": enable_retrieval_augmented,
                    "stylometric": enable_stylometric,
                    "perplexity_optimize": enable_perplexity_optimize,
                    "adversarial": enable_adversarial,
                    "error_injection": enable_error_injection,
                    "detector_optimize": enable_detector_optimize,
                    "classifier": enable_classifier,
                }

                # Build target perplexity profile overrides
                target_profile_overrides = {
                    "target_perplexity_mean": target_perplexity_mean,
                    "target_perplexity_variance": target_perplexity_variance,
                }

                # Get identity and style preferences
                identity = st.session_state.identity
                style_instructions = get_style_instructions()

                # If critic loop is enabled, use CriticLoop
                if enable_critic:
                    def pipeline_factory(intensity):
                        return HumanizationPipeline(
                            intensity=intensity,
                            model=model_choice,
                            api_key=API_KEY,
                            base_url=BASE_URL,
                            stage_overrides=stage_overrides,
                            identity=identity,
                            style_instructions=style_instructions,
                            target_perplexity_profile=target_profile_overrides,
                        )

                    critic = CriticLoop(
                        pipeline_factory=pipeline_factory,
                        max_retries=CRITIC_MAX_RETRIES,
                        risk_threshold=critic_threshold,
                    )

                    with st.spinner("Running critic loop (this may take a moment)..."):
                        result = critic.run(input_text, initial_intensity=rewrite_intensity)

                    final_text = result["final_text"]
                    st.session_state.humanized_output = final_text
                    st.session_state.critic_history = result["attempts"]
                    st.session_state.original_input_text = input_text
                    # Critic loop doesn't expose final_similarity directly;
                    # compute it using the similarity evaluator
                    from humanizer.similarity import SimilarityEvaluator
                    _sim_eval = SimilarityEvaluator()
                    st.session_state.final_similarity = _sim_eval.score(input_text, final_text)

                    # Display attempt progression
                    st.markdown("**Critic Loop Results:**")
                    for i, attempt in enumerate(result["attempts"], 1):
                        is_last = (i == len(result["attempts"]))
                        passed = attempt["risk_score"] <= critic_threshold
                        css_class = "attempt-success" if passed else "attempt-fail"
                        status_label = "(passed)" if passed else "(retry)"
                        if is_last and not passed:
                            status_label = "(best effort)"
                        st.markdown(
                            f'<div class="attempt-box {css_class}">'
                            f'Attempt {i} (intensity {attempt["intensity_used"]}): '
                            f'risk {attempt["risk_score"]}% {status_label}</div>',
                            unsafe_allow_html=True
                        )

                    if result["success"]:
                        st.success("Critic loop succeeded - risk below threshold.")
                    else:
                        st.warning("Critic loop exhausted retries. Showing best result.")

                    st.markdown(final_text)

                else:
                    # Standard pipeline (streaming or sync)
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

                    pipeline = HumanizationPipeline(
                        intensity=rewrite_intensity,
                        model=model_choice,
                        api_key=API_KEY,
                        base_url=BASE_URL,
                        stage_overrides=stage_overrides,
                        progress_callback=update_progress,
                        identity=identity,
                        style_instructions=style_instructions,
                        target_perplexity_profile=target_profile_overrides,
                    )

                    stream_container = st.empty()
                    final_text = ""

                    try:
                        if enable_llm:
                            collected_chunks = []
                            postprocessed_text = None

                            for chunk in pipeline.process_stream(input_text):
                                if chunk == "\n\n__POSTPROCESSED__\n\n":
                                    postprocessed_text = ""
                                    continue
                                if postprocessed_text is not None:
                                    postprocessed_text += chunk
                                    continue
                                collected_chunks.append(chunk)
                                stream_container.markdown(''.join(collected_chunks))

                            if postprocessed_text:
                                final_text = postprocessed_text
                                stream_container.markdown(final_text)
                            else:
                                final_text = ''.join(collected_chunks)
                        else:
                            with st.spinner("Processing through pipeline stages..."):
                                final_text = pipeline.process(input_text)
                            stream_container.markdown(final_text)

                        progress_placeholder.empty()

                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        final_text = ""

                    # Only update session state if the run produced output;
                    # on error (final_text empty), retain the last successful
                    # output and analytics so the UI keeps showing them (Req 12.7).
                    if final_text:
                        st.session_state.humanized_output = final_text
                        st.session_state.final_similarity = pipeline.final_similarity
                        st.session_state.original_input_text = input_text

                # Post-output display (metrics, voice profile, download, feedback)
                if st.session_state.humanized_output:
                    final_text = st.session_state.humanized_output

                    st.markdown("<br>", unsafe_allow_html=True)
                    new_stats = get_text_analytics(final_text)
                    n1, n2, n3, n4, n5 = st.columns(5)
                    n1.metric("Sentences", new_stats['sentences'])
                    n2.metric("Avg Length", f"{new_stats['avg_len']} wds")
                    n3.metric("Grade Level", new_stats['grade'])
                    n4.metric("Vocab Diversity", f"{new_stats['vocabulary_diversity']:.2f}")

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

                    # --- Before/After Detection Risk & Similarity Analytics ---
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("#### Detection & Similarity Analytics")

                    original_for_scoring = st.session_state.original_input_text or input_text
                    before_score, before_source = detection_risk_score(original_for_scoring, classifier=None)
                    after_score, after_source = detection_risk_score(final_text, classifier=None)

                    ana_c1, ana_c2, ana_c3 = st.columns(3)
                    with ana_c1:
                        st.metric(
                            "Detection Risk (Before)",
                            f"{before_score:.0f} / 100",
                            help=f"Scored via {before_source}"
                        )
                    with ana_c2:
                        score_change = after_score - before_score
                        st.metric(
                            "Detection Risk (After)",
                            f"{after_score:.0f} / 100",
                            delta=f"{score_change:+.0f}",
                            delta_color="inverse",
                            help=f"Scored via {after_source}"
                        )
                    with ana_c3:
                        similarity_val = st.session_state.final_similarity
                        if similarity_val is not None:
                            st.metric(
                                "Semantic Similarity",
                                f"{similarity_val:.3f}",
                                help="Cosine similarity between original and final text (0.0-1.0)"
                            )
                        else:
                            st.metric("Semantic Similarity", "N/A")

                    st.caption(
                        "Scores are estimates based on heuristic analysis and may not "
                        "match specific AI detection tools."
                    )

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.download_button(
                        label="Download Humanized Text",
                        data=final_text,
                        file_name="humanized_output.txt",
                        mime="text/plain",
                        use_container_width=True,
                        key="humanize_download"
                    )

                    # Voice Profile Comparison
                    st.markdown("<br>", unsafe_allow_html=True)
                    input_voice = get_voice_profile(input_text)
                    output_voice = get_voice_profile(final_text)
                    display_voice_comparison(input_voice, output_voice)

                    # Revision Feedback Loop
                    display_revision_feedback(final_text, "humanize")

        else:
            # Show existing output if available
            if st.session_state.humanized_output:
                st.markdown(st.session_state.humanized_output)
            else:
                st.info(
                    "Your humanized text will appear here. Adjust settings above "
                    "and click **Humanize Text** to start the pipeline."
                )


# =============================================================================
# CO-WRITE TAB
# =============================================================================
with tab_cowrite:
    st.markdown("### Co-Write Mode")
    st.caption(
        "Generate academic text from your rough notes, bullet points, and ideas. "
        "The AI will build complete prose around your phrasing."
    )

    cw_c1, cw_c2 = st.columns(2)

    with cw_c1:
        cw_intent = st.text_input(
            "Thesis / Intent Statement",
            placeholder="What is the main argument or purpose of this section?",
            key="cw_intent"
        )
        cw_audience = st.text_input(
            "Target Audience",
            placeholder="e.g., Academic reviewers, Graduate students, General public",
            key="cw_audience"
        )

    with cw_c2:
        cw_model = st.selectbox(
            "Model",
            AVAILABLE_MODELS,
            index=0,
            key="cw_model"
        )

    cw_notes = st.text_area(
        "Rough Notes / Ideas",
        height=150,
        placeholder="Paste your bullet points, rough sentences, fragmented ideas here...",
        key="cw_notes"
    )

    cw_arguments = st.text_area(
        "Key Arguments (one per line)",
        height=100,
        placeholder="First argument\nSecond argument\nThird argument",
        key="cw_arguments"
    )

    st.write("")
    trigger_cowrite = st.button("Generate", type="primary", key="cowrite_btn")

    if trigger_cowrite:
        if not cw_intent or not cw_notes:
            st.warning("Please provide at least an intent statement and some rough notes.")
        else:
            identity = st.session_state.identity
            style_instructions = get_style_instructions()

            # Parse arguments (one per line)
            arguments = [a.strip() for a in cw_arguments.split("\n") if a.strip()] if cw_arguments else []

            cowriter = CoWriter(
                model=cw_model,
                api_key=API_KEY,
                base_url=BASE_URL,
                identity=identity,
                style_instructions=style_instructions,
            )

            stream_container = st.empty()
            collected_chunks = []

            try:
                for chunk in cowriter.generate_stream(
                    intent=cw_intent,
                    audience=cw_audience or "Academic audience",
                    notes=cw_notes,
                    arguments=arguments,
                ):
                    collected_chunks.append(chunk)
                    stream_container.markdown(''.join(collected_chunks))

                final_cowrite = ''.join(collected_chunks)
                st.session_state.cowrite_output = final_cowrite

            except Exception as e:
                st.error(f"An error occurred during generation: {str(e)}")
                final_cowrite = ''.join(collected_chunks)
                if final_cowrite:
                    st.session_state.cowrite_output = final_cowrite
                    st.warning("Partial output preserved from before the error.")

            if final_cowrite:
                # Show analytics
                st.markdown("<br>", unsafe_allow_html=True)
                cw_stats = get_text_analytics(final_cowrite)
                s1, s2, s3, s4, s5 = st.columns(5)
                s1.metric("Sentences", cw_stats['sentences'])
                s2.metric("Avg Length", f"{cw_stats['avg_len']} wds")
                s3.metric("Grade Level", cw_stats['grade'])
                s4.metric("Vocab Diversity", f"{cw_stats['vocabulary_diversity']:.2f}")
                s5.metric("AI Risk", f"{cw_stats['ai_risk']}%")

                # Voice profile of generated text
                st.markdown("<br>", unsafe_allow_html=True)
                cw_voice = get_voice_profile(final_cowrite)
                st.markdown("#### Voice Profile of Generated Text")
                vp1, vp2, vp3, vp4, vp5 = st.columns(5)
                vp1.metric("Hedges", cw_voice["hedges"])
                vp2.metric("Boosters", cw_voice["boosters"])
                vp3.metric("Pronouns", cw_voice["pronouns"])
                vp4.metric("Engagement", cw_voice["engagement"])
                vp5.metric("TTR", f"{cw_voice['ttr']:.3f}")

                st.download_button(
                    label="Download Generated Text",
                    data=final_cowrite,
                    file_name="cowrite_output.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key="cowrite_download"
                )

                # Revision Feedback Loop
                display_revision_feedback(final_cowrite, "cowrite")

    else:
        if st.session_state.cowrite_output:
            st.markdown(st.session_state.cowrite_output)
        else:
            st.info(
                "Fill in your notes and arguments above, then click **Generate** "
                "to create academic text from your ideas."
            )
