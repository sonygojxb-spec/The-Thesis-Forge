import streamlit as st
import re
import textstat
import requests
import json

# ==========================================
# 1. PAGE CONFIG & TARGETED CSS
# ==========================================
st.set_page_config(page_title="AI Academic Rewriter", layout="wide", initial_sidebar_state="collapsed")

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
            margin-bottom: 3rem;
            font-weight: 500;
        }

        /* Style the primary text areas to look like Quillbot's input box */
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
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. APP CONFIGURATION & CREDENTIALS
# ==========================================
# Hardcoded API Key (Keep this script private!)
API_KEY = "insert your api key here"
BASE_URL = "https://api.freemodel.dev"

# Header Section
st.markdown("<h1>✨ AI Academic Rewriter</h1>", unsafe_allow_html=True)
st.markdown('<div class="subtitle">Humanize dense academic text, improve flow, and bypass AI detectors.</div>', unsafe_allow_html=True)

# Top Control Bar (Centered)
st.write("")
col_space_left, col_controls, col_space_right = st.columns([1, 4, 1])

with col_controls:
    c1, c2, c3 = st.columns([1.5, 2, 1.5])
    with c1:
        model_choice = st.selectbox("🧠 Model", ["gpt-4o", "gpt-4o-mini", "openai-t1-sg", "openai-t2-sg"], index=0)
    with c2:
        rewrite_intensity = st.slider("🎛️ Stealth Intensity", 1, 5, 4, help="Level 4-5 recommended for bypassing AI detectors.")
    with c3:
        st.write("") # Vertical spacer
        st.write("") # Vertical spacer
        trigger_rewrite = st.button("🚀 Paraphrase Text", type="primary")

st.markdown("<br><br>", unsafe_allow_html=True)

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def fix_encoding_artifacts(input_string):
    replacements = {
        "â€™": "'", "â€”": "—", "â€œ": '"', "â€": '"', 
        "â€˜": "'", "â€¦": "…", "â€“": "–"
    }
    for bad, good in replacements.items():
        input_string = input_string.replace(bad, good)
    return input_string

def get_analytics(input_text):
    if not input_text.strip():
        return {"sentences": 0, "avg_len": 0, "grade": "N/A"}
    sentences = re.split(r'(?<=[.!?])\s+', input_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    avg_len = round(sum(len(s.split()) for s in sentences) / len(sentences), 1) if sentences else 0
    try:
        grade = round(textstat.flesch_kincaid_grade(input_text), 1)
    except Exception:
        grade = "N/A"
    return {"sentences": len(sentences), "avg_len": avg_len, "grade": grade}

def run_ai_rewrite_stream(input_text, intensity, model):
    clean_input = fix_encoding_artifacts(input_text)
    
    intensity_guides = {
        1: "Fix grammatical errors and awkward phrasing, but keep the exact human cadence and vocabulary untouched.",
        2: "Slightly restructure overly dense sentences. Swap out highly predictable AI words for natural phrasing.",
        3: "Moderate editing. Break up dense walls of text. Ensure the flow feels like a human researcher drafting a paper.",
        4: "Strong stealth humanization. Actively maximize burstiness by mixing very short sentences with longer ones. Use high perplexity vocabulary. Sound like a native academic, not a textbook.",
        5: "Maximum stealth rewriting. Thoroughly restructure the text to prioritize a conversational but rigorous academic tone. Avoid all common AI tropes, transition words, and rhythmic sentence structures."
    }
    
    guide = intensity_guides.get(intensity, intensity_guides[4])
    
    system_prompt = (
        "You are an academic researcher writing a draft. Your primary goal is to humanize the text to bypass AI detection "
        "while preserving the exact scientific meaning and domain-specific terminology.\n\n"
        "Strictly adhere to the following rules:\n"
        "1. Maximize Burstiness: Drastically vary sentence lengths. Mix very short, direct sentences with complex ones.\n"
        "2. High Perplexity: Avoid predictable, sterile AI transitions (e.g., 'Moreover', 'Furthermore', 'Crucially', 'Delving into', 'A central challenge'). Use natural, slightly less polished phrasing.\n"
        "3. Tone: Sound like a real human expert drafting a paper. It must be highly readable and dynamic, but strictly maintain rigorous academic formality. Do not use colloquialisms, conversational filler, or casual punchy phrases.\n"
        "4. Do NOT alter scientific facts, specific data points, or core terminology.\n"
        "5. Output ONLY the rewritten text. No preambles, no quotes, no explanations.\n\n"
        f"Intensity Guideline: {guide}"
    )
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    endpoint = f"{BASE_URL.rstrip('/')}/v1/chat/completions"
    calculated_temp = 0.4 + (intensity * 0.12) 
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please rewrite this text to be undetectable by AI:\n\n{clean_input}"}
        ],
        "temperature": min(calculated_temp, 1.2),
        "stream": True
    }
    
    response = requests.post(endpoint, json=payload, headers=headers, stream=True, timeout=30)
    response.raise_for_status()
    
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        delta = chunk["choices"][0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]
                except json.JSONDecodeError:
                    continue

# ==========================================
# 4. MAIN UI LAYOUT (SIDE-BY-SIDE)
# ==========================================
col_left, col_right = st.columns(2, gap="large")

# LEFT COLUMN: INPUT
with col_left:
    st.markdown("### 📝 Original Draft")
    input_text = st.text_area("Original Text", height=350, label_visibility="collapsed", placeholder="Paste your text here to rewrite and humanize...")
    orig_stats = get_analytics(input_text)
    
    if input_text:
        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Sentences", orig_stats['sentences'])
        m2.metric("Avg. Length", f"{orig_stats['avg_len']} wds")
        m3.metric("Grade Level", orig_stats['grade'])

# RIGHT COLUMN: OUTPUT
with col_right:
    st.markdown("### 🎯 Rewritten Output")
    
    if trigger_rewrite:
        if not input_text.strip():
            st.warning("Please paste some text on the left first.")
        else:
            stream_container = st.empty()
            final_rewritten_text = ""
            
            try:
                with st.spinner("Analyzing and rewriting..."):
                    generator = run_ai_rewrite_stream(input_text, rewrite_intensity, model_choice)
                    final_rewritten_text = stream_container.write_stream(generator)
                
                # Show Post-Processing Analytics
                if final_rewritten_text:
                    st.markdown("<br>", unsafe_allow_html=True)
                    new_stats = get_analytics(final_rewritten_text)
                    n1, n2, n3 = st.columns(3)
                    n1.metric("Sentences", new_stats['sentences'])
                    n2.metric("Avg. Length", f"{new_stats['avg_len']} wds")
                    
                    delta = None
                    if isinstance(new_stats['grade'], (int, float)) and isinstance(orig_stats['grade'], (int, float)):
                        delta = round(orig_stats['grade'] - new_stats['grade'], 1)
                        n3.metric("Grade Level", new_stats['grade'], delta=f"{-delta} level(s)" if delta != 0 else None, delta_color="inverse")
                    else:
                        n3.metric("Grade Level", new_stats['grade'])
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.download_button(
                        label="📥 Download Final Draft",
                        data=final_rewritten_text,
                        file_name="humanized_draft.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                    
            except requests.exceptions.HTTPError as e:
                st.error(f"API Error: Please verify your base URL. Details: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
    else:
        st.info("Your humanized text will appear here. Adjust your settings above and click **Paraphrase Text**.")
