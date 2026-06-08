"""
Stage 3: Multi-pass LLM Rewriting

Performs two passes of LLM-based rewriting with different temperatures
and system prompts to create natural-sounding academic text.

Pass 1: Focuses on natural sentence rhythm, varied structure, and authentic voice.
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

    def __init__(self, aggression=0.5, model=None, api_key=None, base_url=None, identity=None,
                 style_instructions=None):
        """
        Args:
            aggression: Float 0-1 controlling rewrite intensity.
            model: Model name to use.
            api_key: API key (defaults to config value).
            base_url: API base URL (defaults to config value).
            identity: Optional AcademicIdentity instance for role conditioning.
            style_instructions: Optional string with style preferences to append to prompts.
        """
        self.aggression = aggression
        self.model = model or DEFAULT_MODEL
        self.api_key = api_key or API_KEY
        self.base_url = base_url or BASE_URL
        self.identity = identity
        self.style_instructions = style_instructions or ""

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

        # Pass 1: Natural rhythm and sentence variation
        try:
            pass1_result = self._llm_pass(
                text,
                system_prompt=self._get_pass1_prompt(),
                temperature=self._get_pass1_temperature(),
                stream_callback=stream_callback,
            )
        except RuntimeError:
            # Pass 1 failed; return original text
            return text

        if not pass1_result.strip():
            return text

        # Pass 2: Academic voice injection (only at higher aggression)
        if self.aggression >= 0.5:
            try:
                pass2_result = self._llm_pass(
                    pass1_result,
                    system_prompt=self._get_pass2_prompt(),
                    temperature=self._get_pass2_temperature(),
                    stream_callback=stream_callback,
                )
                return pass2_result if pass2_result.strip() else pass1_result
            except RuntimeError:
                # Pass 2 failed; return Pass 1 result
                return pass1_result

        return pass1_result

    def pass1_stream(self, text):
        """
        Run Pass 1 and yield chunks for streaming.

        Args:
            text: Input text to rewrite.

        Yields:
            Text chunks from Pass 1.
        """
        if not text.strip():
            yield text
            return

        for chunk in self._llm_pass_stream(
            text,
            system_prompt=self._get_pass1_prompt(),
            temperature=self._get_pass1_temperature(),
        ):
            yield chunk

    def pass2_stream(self, text):
        """
        Run Pass 2 and yield chunks for streaming.

        Args:
            text: Input text (should be Pass 1 output).

        Yields:
            Text chunks from Pass 2.
        """
        if not text.strip():
            yield text
            return

        for chunk in self._llm_pass_stream(
            text,
            system_prompt=self._get_pass2_prompt(),
            temperature=self._get_pass2_temperature(),
        ):
            yield chunk

    def process_stream(self, text):
        """
        Process text and yield chunks for streaming display.
        Kept for backward compatibility.

        Args:
            text: Input text to rewrite.

        Yields:
            Text chunks as they arrive from the LLM.
        """
        if not text.strip():
            yield text
            return

        # Pass 1: Collect internally
        pass1_chunks = []
        try:
            for chunk in self.pass1_stream(text):
                pass1_chunks.append(chunk)
        except RuntimeError:
            yield text
            return

        pass1_result = ''.join(pass1_chunks)

        if not pass1_result.strip():
            yield text
            return

        # Pass 2: Only at higher aggression
        if self.aggression >= 0.5:
            try:
                for chunk in self.pass2_stream(pass1_result):
                    yield chunk
            except RuntimeError:
                # Pass 2 failed; yield Pass 1 result
                yield pass1_result
        else:
            yield pass1_result

    def _get_pass1_prompt(self):
        """System prompt for pass 1: Indian English academic style with natural variation."""
        intensity_detail = ""
        if self.aggression >= 0.7:
            intensity_detail = (
                "Be bold in varying sentence structure. Some sentences must be just 5-8 words. "
                "Others should stretch beyond 35 words with subordinate clauses. "
                "Use unexpected but precise vocabulary choices. Break patterns aggressively. "
            )
        elif self.aggression >= 0.4:
            intensity_detail = (
                "Moderately vary sentence structures. Mix short and long sentences. "
                "Replace overly generic phrasing with more natural Indian English alternatives. "
            )
        else:
            intensity_detail = (
                "Make light adjustments to sentence rhythm. "
                "Fix only the most obviously stiff or formulaic phrasing. "
            )

        return (
            "You are an Indian academic researcher rewriting a draft in your natural voice. "
            "You have studied and published in India and write in Indian English academic style.\n\n"
            + (self.identity.get_prompt_injection() + "\n\n" if self.identity else "")
            + "CRITICAL RULES:\n"
            "1. BRITISH ENGLISH SPELLINGS: Always use British/Indian English spellings - "
            "analyse (not analyze), organise (not organize), behaviour (not behavior), "
            "colour (not color), favour (not favor), programme (not program), "
            "centre (not center), defence (not defense), utilise (not utilize).\n"
            "2. BURSTINESS: Vary sentence lengths dramatically. Some sentences should be very short "
            "(5-10 words). Others should be long and complex (30-40 words) with multiple clauses. "
            "Real humans do not write uniformly.\n"
            "3. INDIAN ENGLISH CONSTRUCTIONS: Use constructions natural to Indian academic English - "
            "use 'one' instead of 'you' for impersonal statements, use characteristic passive voice "
            "constructions ('it was observed that', 'it has been found'), place qualifiers and "
            "adverbial phrases at sentence beginnings.\n"
            "4. VOCABULARY: Avoid ALL typical GPT/AI transition words entirely. Never use: Moreover, "
            "Furthermore, Additionally, Consequently, Nevertheless, It is worth noting, Crucially, "
            "Delving into, Multifaceted, Tapestry, Leveraging, Spearheading, Groundbreaking, "
            "Cutting-edge, Paradigm shift, Synergy, Ecosystem, Seamless, Transformative, "
            "Instrumental, Holistic, Nuanced, Comprehensive understanding.\n"
            "5. NATURAL FLOW: Write like a real Indian researcher drafting - sometimes starting "
            "sentences with 'And' or 'But', using parenthetical asides, varying paragraph lengths. "
            "Occasionally use discourse markers like 'as such', 'in this regard', 'broadly speaking'.\n"
            "6. PRESERVE: Keep all scientific facts, data, terminology, and core meaning intact.\n"
            "7. OUTPUT: Only the rewritten text. No preambles, no explanations, no quotes.\n\n"
            f"{intensity_detail}"
            + (f"\n\n{self.style_instructions}" if self.style_instructions else "")
        )

    def _get_pass2_prompt(self):
        """System prompt for pass 2: Indian English academic voice injection."""
        identity_context = ""
        if self.identity:
            identity_context = self.identity.get_prompt_injection() + "\n\n"

        return (
            "You are polishing an academic text to add authentic Indian English academic voice. "
            "Make subtle but effective adjustments:\n\n"
            + identity_context
            + "1. Add hedging patterns common in Indian academic writing where appropriate: "
            "'it may be observed that', 'one could argue', 'it appears reasonable to suggest', "
            "'it is pertinent to note', 'one may contend that'.\n"
            "2. Use 'shall' instead of 'will' occasionally for formal tone.\n"
            "3. Use characteristic Indian English constructions: 'the same is true for', "
            "'keeping in view', 'with respect to the above', 'in this regard', 'as such'.\n"
            "4. Where relevant, add brief observations (We observed, Our analysis suggests, "
            "In our view, It was noted that).\n"
            "5. Vary formality slightly - some sentences can be slightly less formal than others.\n"
            "6. Ensure transitions between ideas feel natural, not formulaic. Never use AI-typical "
            "transitions (Moreover, Furthermore, Additionally, Consequently).\n"
            "7. Keep British English spellings throughout (analyse, organise, behaviour, colour).\n"
            "8. Keep the text academically rigorous but make it sound like a real Indian researcher wrote it.\n"
            "9. Do NOT add any new information or change scientific facts.\n"
            "10. Output ONLY the polished text. No explanations.\n"
            + (f"\n{self.style_instructions}" if self.style_instructions else "")
        )

    def _get_pass1_temperature(self):
        """Calculate temperature for pass 1."""
        return min(1.4, LLM_PASS1_TEMPERATURE_BASE +
                   self.aggression * LLM_TEMPERATURE_INTENSITY_FACTOR)

    def _get_pass2_temperature(self):
        """Calculate temperature for pass 2."""
        return min(1.4, LLM_PASS2_TEMPERATURE_BASE +
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
                {"role": "user", "content": f"Rewrite this academic text in natural Indian English academic style, varying sentence lengths dramatically and avoiding any AI-typical phrasing:\n\n{text}"}
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
