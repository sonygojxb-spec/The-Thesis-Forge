# ✨ The Thesis Forge

**The Thesis Forge** is an advanced, AI-powered academic text rewriter built for researchers, graduate students, and academics. Designed as a Streamlit web application, this tool goes beyond simple paraphrasing—it strategically "humanizes" dense or machine-generated text to improve natural flow, enhance readability, and confidently bypass AI detectors.

When drafting complex academic documents, it is incredibly easy for text to sound sterile, robotic, or overly polished. The Thesis Forge counters this by maximizing sentence "burstiness" and vocabulary perplexity, ensuring your writing sounds like a rigorous human expert rather than a predictable algorithm.

---

### 🚀 Key Features

* **LLM-Powered Stealth Humanization:** Uses custom prompting to eliminate predictable AI transition words while strictly preserving core scientific facts, data points, and domain-specific terminology.
* **Adjustable Stealth Intensity:** Fine-tune the AI's behavior with a 1-5 slider, ranging from basic grammar correction to maximum structural stealth.
* **Real-Time Text Streaming:** Watch your draft seamlessly transform live via streaming API integration.
* **Integrated Text Analytics:** Instantly compare your original and rewritten drafts using built-in metrics, including sentence count, average word length, and Flesch-Kincaid readability grades.
* **Sleek Dark-Mode UI:** A beautiful, custom-styled interface featuring a distraction-free side-by-side layout and modern, responsive design.

---

### 🛠️ Prerequisites & Installation

To run this application locally, you will need Python installed on your machine. 

**1. Install required dependencies**
```bash
pip install streamlit textstat requests
```
**2. API Key Configuration**
```bash
This app uses the freemodel.dev API. Ensure your valid API key is set within the app.py script.
```
**3. Usage**
```bash
streamlit run app.py
```
