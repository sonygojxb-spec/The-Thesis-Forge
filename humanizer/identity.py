"""
Identity and Role Conditioning Module

Provides academic identity configuration that can be injected into
LLM prompts to condition the rewriting style based on the writer's
role, field, and preferences.
"""

from humanizer.config import ACADEMIC_ROLES, ACADEMIC_FIELDS, STYLE_PREFERENCES


class AcademicIdentity:
    """
    Represents an academic identity for conditioning LLM prompts.

    Attributes:
        role: Academic role (e.g., 'PhD student', 'Professor').
        field: Academic field (e.g., 'Computer Science', 'Physics').
        institution: Optional institution name.
        style_preference: Writing style: 'formal', 'semi-formal', or 'conversational'.
    """

    def __init__(self, role, field, institution=None, style_preference="formal"):
        """
        Args:
            role: Academic role string.
            field: Academic field string.
            institution: Optional institution name.
            style_preference: One of 'formal', 'semi-formal', 'conversational'.
        """
        self.role = role
        self.field = field
        self.institution = institution
        self.style_preference = style_preference if style_preference in STYLE_PREFERENCES else "formal"

    def get_prompt_injection(self):
        """
        Generate a string to inject into LLM system prompts describing this identity.

        Returns:
            str: Identity description for prompt conditioning.
        """
        parts = [
            f"You are writing as a {self.role} in {self.field}."
        ]

        if self.institution:
            parts.append(f"You are affiliated with {self.institution}.")

        if self.style_preference == "formal":
            parts.append(
                "Write in a formal academic register with precise terminology "
                "and structured argumentation typical of the field."
            )
        elif self.style_preference == "semi-formal":
            parts.append(
                "Write in a semi-formal academic style that balances precision "
                "with accessibility, as in a well-written review paper."
            )
        elif self.style_preference == "conversational":
            parts.append(
                "Write in a conversational academic style, as if explaining "
                "to an informed colleague, while maintaining scholarly rigour."
            )

        parts.append(
            f"Use discourse conventions and terminology natural to {self.field}."
        )

        return " ".join(parts)

    def get_discourse_preferences(self):
        """
        Return a list of preferred discourse markers based on field and style.

        Returns:
            list: Discourse markers appropriate for this identity.
        """
        # Base markers for all styles
        base_markers = ["in this context", "as such", "broadly speaking"]

        # Style-specific markers
        style_markers = {
            "formal": [
                "it is pertinent to note",
                "one may observe that",
                "the findings suggest",
                "it has been established that",
                "in view of the above",
            ],
            "semi-formal": [
                "notably",
                "in this regard",
                "one might argue",
                "it appears that",
                "this suggests",
            ],
            "conversational": [
                "interestingly",
                "put simply",
                "the key point here is",
                "what this means is",
                "looking at this closely",
            ],
        }

        # Field-specific markers
        field_markers = {
            "Computer Science": ["algorithmically", "in terms of complexity", "empirically"],
            "Physics": ["experimentally", "theoretically", "in the classical limit"],
            "Biology": ["mechanistically", "in vivo", "phenotypically"],
            "Chemistry": ["stoichiometrically", "kinetically", "thermodynamically"],
            "Mathematics": ["by construction", "without loss of generality", "trivially"],
            "Psychology": ["behaviourally", "cognitively", "in terms of affect"],
            "Economics": ["ceteris paribus", "marginally", "in equilibrium"],
            "Literature": ["textually", "narratively", "in terms of discourse"],
        }

        markers = list(base_markers)
        markers.extend(style_markers.get(self.style_preference, []))
        markers.extend(field_markers.get(self.field, []))

        return markers
