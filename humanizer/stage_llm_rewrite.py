"""
Stage 3: Multi-pass LLM Rewriting

Performs two passes of LLM-based rewriting with different temperatures
and system prompts to create natural-sounding academic text.

Pass 1: Focuses on natural sentence rhythm and removing AI patterns.
Pass 2: Injects personal academic voice with hedging and discipline phrasing.
"""

import json
import requests

from humanizer.config import (
    API_KEY, BASE_URL, DEFAULT_MODEL,
    LLM_PASS1_TEMPERATURE_BASE, LLM_PASS2_TEMPERATURE_BASE,
    LLM_TEMPERATURE_INTENSITY_FACTOR,
)


class LLMRewriter:
    """Multi-pass LLM rewriting with streaming support."""

    def __init__(self, aggression=0.5, model=None, api_key=None, base_url=None):
        """
        Args:
            aggression: Float 0-1 controlling rewrite intensity.
            model: Model name to use.
            api_key: API key (defaults to config value).
            base_url: API base URL (defaults to config value).
        """
        self.aggression = aggression
        self.model = model or DEFAULT_MODEL
        self.api_key = api_key or API_KEY
        self.base_url = base_url or BASE_URL

    def process(self, text, stream_callback=None):
        """
        Apply multi-pass LLM rewriting.

        Args:
            text: Input text to rewrite.
            stream_callback: Optional callback(chunk) for streaming output.

        Returns:
            Rewritten text.
        """
        if not text.strip():
            return text

        # Pass 1: Natural rhythm and de-AI-ification
        pass1_result = self._llm_pass(
            text,
            system_prompt=self._get_pass1_prompt(),
            temperature=self._get_pass1_temperature(),
            stream_callback=stream_callback,
        )

        if not pass1_result.strip():
            return text

        # Pass 2: Academic voice injection (only at higher aggression)
        if self.aggression >= 0.5:
            pass2_result = self._llm_pass(
                pass1_result,
                system_prompt=self._get_pass2_prompt(),
                temperature=self._get_pass2_temperature(),
                stream_callback=stream_callback,
            )
            return pass2_result if pass2_result.strip() else pass1_result

        return pass1_result

    def process_stream(self, text):
        """
        Process text and yield chunks for streaming display.

        Args:
            text: Input text to rewrite.

        Yields:
            Text chunks as they arrive from the LLM.
        """
        if not text.strip():
            yield text
            return

        # Pass 1: Stream output
        pass1_chunks = []
        for chunk in self._llm_pass_stream(
            text,
            system_prompt=self._get_pass1_prompt(),
            temperature=self._get_pass1_temperature(),
        ):
            pass1_chunks.append(chunk)
            yield chunk

        pass1_result = ''.join(pass1_chunks)

        # Pass 2: Only at higher aggression
        if self.aggression >= 0.5 and pass1_result.strip():
            yield "\n\n---PASS2---\n\n"
            for chunk in self._llm_pass_stream(
                pass1_result,
                system_prompt=self._get_pass2_prompt(),
                temperature=self._get_pass2_temperature(),
            ):
                yield chunk

    def _get_pass1_prompt(self):
        """System prompt for pass 1: rhythm and de-AI."""
        intensity_detail = ""
        if self.aggression >= 0.7:
            intensity_detail = (
                "Be aggressive in breaking AI patterns. Vary sentence length dramatically. "
                "Some sentences should be 5-8 words. Others 30+. "
                "Use unusual but accurate word choices. "
            )
        elif self.aggression >= 0.4:
            intensity_detail = (
                "Moderately vary sentence structures. Mix short and long sentences. "
                "Replace common AI phrasing with more natural alternatives. "
            )
        else:
            intensity_detail = (
                "Make light adjustments to sentence rhythm. "
                "Fix only the most obvious AI patterns. "
            )

        return (
            "You are an experienced academic researcher rewriting a draft to sound naturally human-written. "
            "Your goal is to make this text undetectable by AI detection tools like Turnitin.\n\n"
            "CRITICAL RULES:\n"
            "1. BURSTINESS: Vary sentence lengths dramatically. Mix very short sentences (5-10 words) "
            "with complex ones (25-40 words). Real humans do not write uniformly.\n"
            "2. PERPLEXITY: Avoid the most predictable word choices. Use slightly unexpected but "
            "accurate vocabulary. Do not use: Moreover, Furthermore, Additionally, Consequently, "
            "Nevertheless, It is worth noting, Crucially, Delving into, Multifaceted, Tapestry.\n"
            "3. NATURAL FLOW: Write like a real researcher drafting - sometimes starting sentences "
            "with 'And' or 'But', occasionally using parenthetical asides, varying paragraph lengths.\n"
            "4. PRESERVE: Keep all scientific facts, data, terminology, and core meaning intact.\n"
            "5. OUTPUT: Only the rewritten text. No preambles, no explanations, no quotes.\n\n"
            f"{intensity_detail}"
        )

    def _get_pass2_prompt(self):
        """System prompt for pass 2: academic voice injection."""
        return (
            "You are polishing an academic text to add authentic human voice. "
            "Make subtle adjustments:\n\n"
            "1. Add occasional hedging language where appropriate (seems to, appears to, "
            "might suggest, could indicate).\n"
            "2. Where relevant, add brief first-person observations (We observed, Our analysis, "
            "In our view).\n"
            "3. Vary formality slightly - some sentences can be slightly less formal than others.\n"
            "4. Ensure transitions between ideas feel natural, not formulaic.\n"
            "5. Keep the text academically rigorous but make it sound like a real person wrote it.\n"
            "6. Do NOT add any new information or change scientific facts.\n"
            "7. Output ONLY the polished text. No explanations.\n"
        )

    def _get_pass1_temperature(self):
        """Calculate temperature for pass 1."""
        return min(1.2, LLM_PASS1_TEMPERATURE_BASE +
                   self.aggression * LLM_TEMPERATURE_INTENSITY_FACTOR)

    def _get_pass2_temperature(self):
        """Calculate temperature for pass 2."""
        return min(1.2, LLM_PASS2_TEMPERATURE_BASE +
                   self.aggression * LLM_TEMPERATURE_INTENSITY_FACTOR)

    def _llm_pass(self, text, system_prompt, temperature, stream_callback=None):
        """Execute a single LLM pass."""
        chunks = []
        for chunk in self._llm_pass_stream(text, system_prompt, temperature):
            chunks.append(chunk)
            if stream_callback:
                stream_callback(chunk)
        return ''.join(chunks)

    def _llm_pass_stream(self, text, system_prompt, temperature):
        """Execute a single LLM pass with streaming."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        endpoint = f"{self.base_url.rstrip('/')}/v1/chat/completions"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Rewrite this text:\n\n{text}"}
            ],
            "temperature": temperature,
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
