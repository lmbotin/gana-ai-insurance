"""
Text extraction module for Track & Trace / AI Operational Liability claims.

Extracts structured information from incident descriptions using LLMs.
"""

import json
import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional

from .config import ExtractionConfig
from .schema import AssetType, ImpactSeverity, IncidentType, SourceModality


class TextExtractor(ABC):
    """Base class for text extraction."""

    @abstractmethod
    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extract structured information from text.

        Args:
            text: Raw text description of the operational incident

        Returns:
            Dictionary with extracted fields and metadata:
            {
                'incident_date': str | None,
                'incident_date_confidence': float,
                'incident_location': str | None,
                'incident_location_confidence': float,
                'incident_description': str | None,
                'incident_description_confidence': float,
                'incident_type': str,
                'incident_type_confidence': float,
                'asset_type': str,
                'asset_type_confidence': float,
                'system_component': str | None,
                'system_component_confidence': float,
                'estimated_liability_cost': float | None,
                'estimated_liability_cost_confidence': float,
                'impact_severity': str,
                'impact_severity_confidence': float,
                'extraction_time_ms': float
            }
        """
        pass


class LLMTextExtractor(TextExtractor):
    """LLM-based text extraction (Claude or OpenAI)."""

    def __init__(self, config: ExtractionConfig):
        """Initialize with configuration."""
        self.config = config

        # Import appropriate client
        if config.llm_provider == "claude":
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=config.api_key)
            except ImportError:
                raise ImportError(
                    "anthropic package required for Claude. "
                    "Install with: pip install anthropic"
                )
        elif config.llm_provider == "openai":
            try:
                import openai
                self.client = openai.OpenAI(api_key=config.api_key)
            except ImportError:
                raise ImportError(
                    "openai package required for OpenAI. "
                    "Install with: pip install openai"
                )
        else:
            raise ValueError(f"Unsupported LLM provider: {config.llm_provider}")

    def _build_prompt(self, text: str) -> str:
        """Build extraction prompt for Track & Trace / AI liability incidents."""
        return f"""You are a liability claims analyst specializing in AI-powered logistics and Track & Trace systems. Extract ONLY information that is explicitly stated in the incident description below.

INCIDENT DESCRIPTION:
{text}

EXTRACTION RULES (CRITICAL - READ CAREFULLY):
1. NEVER infer or guess information not explicitly stated
2. If a field is not mentioned, set value to null and confidence to 0.0
3. If a field is ambiguous or partially mentioned, use low confidence (0.1-0.4)
4. If a field is clearly stated, use high confidence (0.7-0.95)
5. Reserve confidence 0.95+ only for verbatim quotes or explicit statements
6. When in doubt, use "unknown" for enums - DO NOT GUESS

Return ONLY a valid JSON object with this exact structure:
{{
  "incident_date": "ISO datetime string or null",
  "incident_date_confidence": 0.0-1.0,
  "incident_location": "system node, hub ID, route, or facility identifier - or null if not stated",
  "incident_location_confidence": 0.0-1.0,
  "incident_description": "factual summary of what happened - or null",
  "incident_description_confidence": 0.0-1.0,
  "incident_type": "misroute|delay|loss|data_error|prediction_failure|system_outage|other|unknown",
  "incident_type_confidence": 0.0-1.0,
  "asset_type": "shipment|package|container|ai_model|sensor|route|prediction|document|other|unknown",
  "asset_type_confidence": 0.0-1.0,
  "system_component": "specific system or subsystem affected (e.g., 'routing-engine', 'prediction-service') - or null",
  "system_component_confidence": 0.0-1.0,
  "estimated_liability_cost": number or null (ONLY if explicitly stated with currency amount),
  "estimated_liability_cost_confidence": 0.0-1.0,
  "impact_severity": "minor|moderate|severe|critical|unknown",
  "impact_severity_confidence": 0.0-1.0
}}

INCIDENT TYPE DEFINITIONS (use these to classify):
- misroute: Shipment/package sent to wrong destination
- delay: Delivery exceeded SLA or expected timeframe
- loss: Shipment/package/data lost entirely, unrecoverable
- data_error: Incorrect data entry, corrupted records, wrong information processed
- prediction_failure: AI/ML model produced incorrect forecast, recommendation, or classification
- system_outage: System unavailability that caused operational impact
- other: Incident doesn't fit above categories but is clearly described
- unknown: Cannot determine incident type from description

ASSET TYPE DEFINITIONS:
- shipment: Full consignment or shipment
- package: Individual package or parcel
- container: Shipping container
- ai_model: AI/ML model, algorithm, or automated decision system
- sensor: IoT device, tracker, or monitoring equipment
- route: Delivery route, path, or logistics plan
- prediction: AI-generated forecast, ETA, or recommendation
- document: Manifest, shipping document, or record
- other/unknown: Use when asset type is unclear

CONFIDENCE CALIBRATION:
- 0.0: Field not mentioned at all
- 0.1-0.3: Vaguely implied, highly uncertain
- 0.4-0.6: Partially mentioned, some ambiguity
- 0.7-0.85: Clearly stated but not verbatim
- 0.86-0.95: Explicitly stated, minimal ambiguity
- 0.96-1.0: Verbatim quote or unambiguous explicit statement

Return ONLY the JSON object. No explanations, no markdown formatting."""

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response and extract JSON."""
        # Try to find JSON in response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        else:
            raise ValueError(f"No JSON found in LLM response: {response}")

    def extract(self, text: str) -> Dict[str, Any]:
        """Extract structured information using LLM."""
        start_time = datetime.utcnow()

        prompt = self._build_prompt(text)

        try:
            if self.config.llm_provider == "claude":
                response = self.client.messages.create(
                    model=self.config.llm_model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}]
                )
                result_text = response.content[0].text
            else:  # openai
                response = self.client.chat.completions.create(
                    model=self.config.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                result_text = response.choices[0].message.content

            # Parse response
            extracted = self._parse_llm_response(result_text)

            # Add extraction time
            end_time = datetime.utcnow()
            extracted['extraction_time_ms'] = (end_time - start_time).total_seconds() * 1000

            return extracted

        except Exception as e:
            # Return safe default on error
            return self._get_default_extraction(str(e))

    def _get_default_extraction(self, error_msg: str = "") -> Dict[str, Any]:
        """Return default extraction on error - all fields unknown/null with zero confidence."""
        return {
            'incident_date': None,
            'incident_date_confidence': 0.0,
            'incident_location': None,
            'incident_location_confidence': 0.0,
            'incident_description': None,
            'incident_description_confidence': 0.0,
            'incident_type': 'unknown',
            'incident_type_confidence': 0.0,
            'asset_type': 'unknown',
            'asset_type_confidence': 0.0,
            'system_component': None,
            'system_component_confidence': 0.0,
            'estimated_liability_cost': None,
            'estimated_liability_cost_confidence': 0.0,
            'impact_severity': 'unknown',
            'impact_severity_confidence': 0.0,
            'extraction_time_ms': 0.0,
            'error': error_msg
        }


class MockTextExtractor(TextExtractor):
    """Mock extractor for testing (deterministic, no API calls)."""

    def extract(self, text: str) -> Dict[str, Any]:
        """Extract using simple heuristics for T&T / AI liability domain."""
        start_time = datetime.utcnow()

        # Default: all unknown with zero/low confidence (hallucination-avoidance default)
        extracted = {
            'incident_date': None,
            'incident_date_confidence': 0.0,
            'incident_location': None,
            'incident_location_confidence': 0.0,
            'incident_description': text if text else None,
            'incident_description_confidence': 0.9 if text else 0.0,
            'incident_type': 'unknown',
            'incident_type_confidence': 0.0,
            'asset_type': 'unknown',
            'asset_type_confidence': 0.0,
            'system_component': None,
            'system_component_confidence': 0.0,
            'estimated_liability_cost': None,
            'estimated_liability_cost_confidence': 0.0,
            'impact_severity': 'unknown',
            'impact_severity_confidence': 0.0,
        }

        if not text:
            end_time = datetime.utcnow()
            extracted['extraction_time_ms'] = (end_time - start_time).total_seconds() * 1000
            return extracted

        text_lower = text.lower()

        # Heuristic incident type detection
        if 'misroute' in text_lower or 'wrong destination' in text_lower or 'sent to wrong' in text_lower:
            extracted['incident_type'] = 'misroute'
            extracted['incident_type_confidence'] = 0.8
        elif 'delay' in text_lower or 'late' in text_lower or 'missed deadline' in text_lower or 'sla' in text_lower:
            extracted['incident_type'] = 'delay'
            extracted['incident_type_confidence'] = 0.8
        elif 'lost' in text_lower or 'missing' in text_lower or 'cannot locate' in text_lower:
            extracted['incident_type'] = 'loss'
            extracted['incident_type_confidence'] = 0.8
        elif 'data error' in text_lower or 'incorrect data' in text_lower or 'wrong information' in text_lower or 'corrupted' in text_lower:
            extracted['incident_type'] = 'data_error'
            extracted['incident_type_confidence'] = 0.8
        elif 'prediction' in text_lower or 'forecast' in text_lower or 'misclassif' in text_lower or 'model' in text_lower:
            extracted['incident_type'] = 'prediction_failure'
            extracted['incident_type_confidence'] = 0.7
        elif 'outage' in text_lower or 'down' in text_lower or 'unavailable' in text_lower or 'offline' in text_lower:
            extracted['incident_type'] = 'system_outage'
            extracted['incident_type_confidence'] = 0.8

        # Heuristic asset type detection
        if 'shipment' in text_lower or 'consignment' in text_lower:
            extracted['asset_type'] = 'shipment'
            extracted['asset_type_confidence'] = 0.8
        elif 'package' in text_lower or 'parcel' in text_lower:
            extracted['asset_type'] = 'package'
            extracted['asset_type_confidence'] = 0.8
        elif 'container' in text_lower:
            extracted['asset_type'] = 'container'
            extracted['asset_type_confidence'] = 0.8
        elif 'model' in text_lower or 'algorithm' in text_lower or 'ai' in text_lower or 'ml' in text_lower:
            extracted['asset_type'] = 'ai_model'
            extracted['asset_type_confidence'] = 0.7
        elif 'sensor' in text_lower or 'tracker' in text_lower or 'iot' in text_lower:
            extracted['asset_type'] = 'sensor'
            extracted['asset_type_confidence'] = 0.8
        elif 'route' in text_lower or 'path' in text_lower:
            extracted['asset_type'] = 'route'
            extracted['asset_type_confidence'] = 0.7
        elif 'prediction' in text_lower or 'forecast' in text_lower or 'eta' in text_lower:
            extracted['asset_type'] = 'prediction'
            extracted['asset_type_confidence'] = 0.7
        elif 'document' in text_lower or 'manifest' in text_lower or 'record' in text_lower:
            extracted['asset_type'] = 'document'
            extracted['asset_type_confidence'] = 0.8

        # Heuristic severity detection
        if 'critical' in text_lower or 'urgent' in text_lower or 'emergency' in text_lower:
            extracted['impact_severity'] = 'critical'
            extracted['impact_severity_confidence'] = 0.8
        elif 'severe' in text_lower or 'major' in text_lower or 'significant' in text_lower:
            extracted['impact_severity'] = 'severe'
            extracted['impact_severity_confidence'] = 0.7
        elif 'moderate' in text_lower or 'medium' in text_lower:
            extracted['impact_severity'] = 'moderate'
            extracted['impact_severity_confidence'] = 0.7
        elif 'minor' in text_lower or 'small' in text_lower or 'slight' in text_lower:
            extracted['impact_severity'] = 'minor'
            extracted['impact_severity_confidence'] = 0.7

        # System component detection (look for common patterns)
        component_patterns = [
            (r'routing[- ]?engine', 'routing-engine'),
            (r'prediction[- ]?service', 'prediction-service'),
            (r'tracking[- ]?system', 'tracking-system'),
            (r'sorting[- ]?system', 'sorting-system'),
            (r'api[- ]?gateway', 'api-gateway'),
            (r'data[- ]?pipeline', 'data-pipeline'),
            (r'warehouse[- ]?management', 'warehouse-management'),
        ]
        for pattern, component in component_patterns:
            if re.search(pattern, text_lower):
                extracted['system_component'] = component
                extracted['system_component_confidence'] = 0.8
                break

        # Location detection (hub IDs, facility codes)
        location_match = re.search(r'\b(HUB-[A-Z]{2,4}-\d{1,3}|[A-Z]{2,4}-\d{3,6})\b', text, re.IGNORECASE)
        if location_match:
            extracted['incident_location'] = location_match.group(1).upper()
            extracted['incident_location_confidence'] = 0.85

        # Try to extract cost (only if explicitly stated with currency)
        cost_match = re.search(r'\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', text)
        if cost_match:
            cost_str = cost_match.group(1).replace(',', '')
            try:
                extracted['estimated_liability_cost'] = float(cost_str)
                extracted['estimated_liability_cost_confidence'] = 0.7
            except ValueError:
                pass

        end_time = datetime.utcnow()
        extracted['extraction_time_ms'] = (end_time - start_time).total_seconds() * 1000

        return extracted


def create_text_extractor(config: Optional[ExtractionConfig] = None) -> TextExtractor:
    """Factory function to create appropriate text extractor."""
    if config is None:
        from .config import DEFAULT_CONFIG
        config = DEFAULT_CONFIG

    if config.llm_provider == "mock":
        return MockTextExtractor()
    else:
        return LLMTextExtractor(config)
