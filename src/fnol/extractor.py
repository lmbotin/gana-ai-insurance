"""
LLM-based operational liability claim field extraction from conversation transcripts.

Uses OpenAI GPT-4o to extract structured claim information from user utterances.
Adapted for OperationalLiabilityClaim schema.
"""

import json
import logging
from typing import Any, Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


EXTRACTION_SYSTEM_PROMPT = """You are an insurance claim data extraction assistant. Your task is to extract structured information from a caller's statement during an operational liability claim call (AI logistics / Track & Trace).

IMPORTANT RULES:
1. Extract ONLY information explicitly stated by the caller
2. Do NOT infer, assume, or hallucinate any values
3. Use null for any field not explicitly mentioned
4. Preserve the caller's exact wording for descriptions
5. For ambiguous values, use the closest matching enum value or "unknown"

FIELD DEFINITIONS:

Claimant Info:
- claimant.name: The company name (policyholder)
- claimant.policy_number: Insurance policy number
- claimant.contact_phone: Phone number
- claimant.contact_email: Email address

Incident Info:
- incident.incident_type: One of: "misroute", "delay", "loss", "data_error", "prediction_failure", "pricing_error", "system_outage", "other", "unknown"
- incident.incident_date: When the incident occurred (ISO format if possible, or description)
- incident.incident_location: System/node/route where incident occurred (e.g., "HUB-LAX-03", "routing-engine-v2")
- incident.incident_description: What happened (caller's description)

Operational Impact:
- operational_impact.asset_type: One of: "shipment", "package", "container", "ai_model", "sensor", "route", "prediction", "document", "other", "unknown"
- operational_impact.system_component: Specific system/subsystem affected (e.g., "routing-engine", "prediction-service")
- operational_impact.impact_severity: One of: "minor", "moderate", "severe", "critical", "unknown"
- operational_impact.estimated_liability_cost: Numeric estimate if provided (just the number)

Return a JSON object with ONLY the fields that can be extracted from the caller's statement.
Use dot notation for field names (e.g., "claimant.name", "incident.incident_type").
Do NOT include fields that were not mentioned.
"""


class OperationalClaimExtractor:
    """
    Extracts operational liability claim fields from conversation transcripts using GPT-4o.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        """
        Initialize the extractor.

        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
            model: Model to use for extraction (default: gpt-4o)
        """
        self.client = AsyncOpenAI(api_key=api_key) if api_key else AsyncOpenAI()
        self.model = model

    async def extract(
        self,
        transcript: str,
        current_state: Optional[dict] = None,
        conversation_context: Optional[list[dict]] = None,
    ) -> dict[str, Any]:
        """
        Extract operational liability claim fields from a transcript.

        Args:
            transcript: The user's latest utterance
            current_state: Current claim state (to avoid re-extracting known fields)
            conversation_context: Recent conversation turns for context

        Returns:
            Dictionary with extracted field values (dot-notation keys)
        """
        # Build the user message with context
        user_content = self._build_extraction_prompt(
            transcript, current_state, conversation_context
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,  # Low temperature for more deterministic extraction
                max_tokens=1000,
            )

            result_text = response.choices[0].message.content
            extracted = json.loads(result_text) if result_text else {}

            # Validate and clean the extracted data
            cleaned = self._clean_extraction(extracted)

            logger.info(f"Extracted {len(cleaned)} fields from transcript")
            logger.debug(f"Extracted data: {cleaned}")

            return cleaned

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse extraction response as JSON: {e}")
            return {}
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return {}

    def _build_extraction_prompt(
        self,
        transcript: str,
        current_state: Optional[dict],
        conversation_context: Optional[list[dict]],
    ) -> str:
        """Build the prompt for extraction."""
        parts = []

        # Add context about what's already known
        if current_state:
            known_fields = self._get_non_null_fields(current_state)
            if known_fields:
                parts.append("ALREADY COLLECTED (do not re-extract unless corrected):")
                for field, value in known_fields.items():
                    parts.append(f"  - {field}: {value}")
                parts.append("")

        # Add recent conversation context
        if conversation_context:
            parts.append("RECENT CONVERSATION:")
            for turn in conversation_context[-4:]:  # Last 4 turns
                role = turn.get("role", "unknown")
                content = turn.get("content", "")
                parts.append(f"  {role.upper()}: {content}")
            parts.append("")

        # Add the current transcript
        parts.append("CALLER'S LATEST STATEMENT:")
        parts.append(transcript)
        parts.append("")
        parts.append("Extract all claim-related information from the caller's statement above.")

        return "\n".join(parts)

    def _get_non_null_fields(self, state: dict, prefix: str = "") -> dict[str, Any]:
        """Get all non-null fields from a nested state dict."""
        fields = {}

        for key, value in state.items():
            # Skip internal metadata
            if key.startswith("_"):
                continue

            full_key = f"{prefix}.{key}" if prefix else key

            if value is None:
                continue
            elif isinstance(value, dict):
                fields.update(self._get_non_null_fields(value, full_key))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        fields.update(self._get_non_null_fields(item, f"{full_key}.{i}"))
                    elif item is not None:
                        fields[f"{full_key}.{i}"] = item
            else:
                # Skip metadata and default enum values
                if key in ("claim_id", "created_at", "schema_version"):
                    continue
                if value == "unknown":
                    continue
                fields[full_key] = value

        return fields

    def _flatten_dict(self, d: dict, parent_key: str = "") -> dict[str, Any]:
        """Flatten a nested dictionary to dot-notation keys."""
        items = {}
        for key, value in d.items():
            new_key = f"{parent_key}.{key}" if parent_key else key
            if isinstance(value, dict):
                items.update(self._flatten_dict(value, new_key))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        items.update(self._flatten_dict(item, f"{new_key}.{i}"))
                    elif item is not None:
                        items[f"{new_key}.{i}"] = item
            elif value is not None:
                items[new_key] = value
        return items

    def _clean_extraction(self, extracted: dict) -> dict[str, Any]:
        """Clean and validate extracted data."""
        # First, flatten nested dicts to dot-notation keys
        flattened = self._flatten_dict(extracted)

        cleaned = {}

        # Valid enum values for validation
        valid_incident_types = {"misroute", "delay", "loss", "data_error", "prediction_failure",
                               "pricing_error", "system_outage", "other", "unknown"}
        valid_asset_types = {"shipment", "package", "container", "ai_model", "sensor",
                           "route", "prediction", "document", "other", "unknown"}
        valid_severities = {"minor", "moderate", "severe", "critical", "unknown"}

        for key, value in flattened.items():
            if value is None:
                continue

            # Normalize string values
            if isinstance(value, str):
                value = value.strip()
                if not value:
                    continue

                # Validate enum fields
                lower_value = value.lower()

                if key == "incident.incident_type":
                    if lower_value not in valid_incident_types:
                        value = "unknown"
                    else:
                        value = lower_value

                elif key == "operational_impact.asset_type":
                    if lower_value not in valid_asset_types:
                        value = "unknown"
                    else:
                        value = lower_value

                elif key == "operational_impact.impact_severity":
                    if lower_value not in valid_severities:
                        value = "unknown"
                    else:
                        value = lower_value

            # Validate estimated_liability_cost as number
            if key == "operational_impact.estimated_liability_cost" and value:
                try:
                    # Remove currency symbols and commas
                    if isinstance(value, str):
                        value = value.replace("$", "").replace(",", "").strip()
                    value = float(value)
                    if value < 0:
                        continue  # Invalid negative cost
                except (ValueError, TypeError):
                    continue

            cleaned[key] = value

        return cleaned

    async def extract_with_confirmation(
        self,
        transcript: str,
        current_state: Optional[dict] = None,
    ) -> tuple[dict[str, Any], list[str]]:
        """
        Extract fields and identify values that should be confirmed.

        Returns:
            Tuple of (extracted_fields, fields_to_confirm)
        """
        extracted = await self.extract(transcript, current_state)

        # Identify critical fields that should be confirmed
        confirm_fields = []
        critical_keys = [
            "claimant.policy_number",
            "incident.incident_date",
            "claimant.contact_phone",
            "claimant.contact_email",
            "operational_impact.estimated_liability_cost",
        ]

        for key in critical_keys:
            if key in extracted:
                confirm_fields.append(key)

        return extracted, confirm_fields


# Backwards compatibility aliases
PropertyClaimExtractor = OperationalClaimExtractor
FNOLExtractor = OperationalClaimExtractor
