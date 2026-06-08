"""
Co-writing Module

Generates academic text around user-provided notes, ideas, and bullet points.
Uses the same LLM endpoint as the rewriter but with prompts designed for
text generation from rough notes rather than rewriting existing text.
"""

import json
import requests

from humanizer.config import API_KEY, BASE_URL, DEFAULT_MODEL


class CoWriter:
    """Generates academic text from user notes/ideas using LLM calls with streaming."""

    def __init__(self, model=None, api_key=None, base_url=None, identity=None,
                 style_instructions=None):
        """
        Args:
            model: Model name to use.
            api_key: API key (defaults to config value).
            base_url: API base URL (defaults to config value).
            identity: Optional AcademicIdentity instance for role conditioning.
            style_instructions: Optional string with style preferences to append to prompts.
        """
        self.model = model or DEFAULT_MODEL
        self.api_key = api_key or API_KEY
        self.base_url = base_url or BASE_URL
        self.identity = identity
        self.style_instructions = style_instructions or ""

    def generate(self, intent, audience, notes, arguments, stream_callback=None):
        """
        Generate academic text from user-provided notes and structure.

        Args:
            intent: The thesis statement or intent of the text.
            audience: Target audience description.
            notes: Rough notes, bullet points, or ideas from the user.
            arguments: List of key arguments to structure around.
            stream_callback: Optional callback(chunk) for streaming output.

        Returns:
            Generated academic text as a string.
        """
        system_prompt = self._build_system_prompt()
        user_message = self._build_user_message(intent, audience, notes, arguments)

        chunks = []
        for chunk in self._call_llm_stream(system_prompt, user_message):
            chunks.append(chunk)
            if stream_callback:
                stream_callback(chunk)
        return ''.join(chunks)

    def generate_stream(self, intent, audience, notes, arguments):
        """
        Generate academic text and yield chunks for streaming display.

        Args:
            intent: The thesis statement or intent of the text.
            audience: Target audience description.
            notes: Rough notes, bullet points, or ideas from the user.
            arguments: List of key arguments to structure around.

        Yields:
            Text chunks as they arrive from the LLM.
        """
        system_prompt = self._build_system_prompt()
        user_message = self._build_user_message(intent, audience, notes, arguments)

        for chunk in self._call_llm_stream(system_prompt, user_message):
            yield chunk

    def _build_system_prompt(self):
        """Build the system prompt for co-writing."""
        identity_context = ""
        if self.identity:
            identity_context = self.identity.get_prompt_injection() + "\n\n"

        return (
            "You are an academic writing assistant helping a researcher develop complete "
            "academic text from their rough notes and bullet points. "
            "You write in Indian English academic style by default.\n\n"
            + identity_context
            + "CRITICAL RULES:\n"
            "1. BUILD AROUND THE USER'S PHRASING: The user has provided rough notes and key phrases. "
            "Weave their exact phrasing into the generated text wherever possible. Do not paraphrase "
            "their words unnecessarily - instead, build complete academic paragraphs around them.\n"
            "2. PRESERVE ORIGINAL VOICE: The user's word choices, expressions, and phrasing are "
            "intentional. Incorporate them naturally into flowing academic prose.\n"
            "3. BRITISH ENGLISH SPELLINGS: Always use British/Indian English spellings - "
            "analyse (not analyze), organise (not organize), behaviour (not behavior), "
            "colour (not color), favour (not favor), centre (not center).\n"
            "4. ACADEMIC TONE: Maintain a scholarly tone appropriate for the target audience. "
            "Use hedging where appropriate ('it appears', 'one might argue', 'it is pertinent to note').\n"
            "5. LOGICAL STRUCTURE: Organise the output logically based on the arguments provided. "
            "Each argument should form a coherent section or paragraph with smooth transitions.\n"
            "6. NATURAL FLOW: Write like a real Indian researcher - vary sentence lengths, "
            "use parenthetical asides occasionally, start some sentences with discourse markers "
            "like 'as such', 'in this regard', 'broadly speaking'.\n"
            "7. AVOID AI LANGUAGE: Never use: Moreover, Furthermore, Additionally, Consequently, "
            "Nevertheless, Crucially, Delving into, Multifaceted, Tapestry, Leveraging, "
            "Spearheading, Groundbreaking, Cutting-edge, Paradigm shift, Synergy, Seamless, "
            "Transformative, Holistic, Nuanced, Comprehensive understanding.\n"
            "8. OUTPUT: Only the generated academic text. No preambles, no explanations, no meta-commentary.\n"
            + (f"\n{self.style_instructions}" if self.style_instructions else "")
        )

    def _build_user_message(self, intent, audience, notes, arguments):
        """Build the structured user message for the LLM."""
        arguments_text = ""
        if arguments:
            arguments_text = "\n".join(f"- {arg}" for arg in arguments)
        else:
            arguments_text = "(No specific arguments provided)"

        return (
            f"INTENT/THESIS STATEMENT:\n{intent}\n\n"
            f"TARGET AUDIENCE:\n{audience}\n\n"
            f"ROUGH NOTES AND IDEAS:\n{notes}\n\n"
            f"KEY ARGUMENTS TO STRUCTURE AROUND:\n{arguments_text}\n\n"
            "Please generate complete academic text that weaves around my notes and phrasing, "
            "structuring the output logically based on the key arguments provided."
        )

    def _call_llm_stream(self, system_prompt, user_message):
        """Execute LLM call with streaming, yielding text chunks."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        endpoint = f"{self.base_url.rstrip('/')}/v1/chat/completions"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.8,
            "stream": True
        }

        try:
            response = requests.post(
                endpoint, json=payload, headers=headers,
                stream=True, timeout=60
            )
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
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"LLM API error: {e}")
