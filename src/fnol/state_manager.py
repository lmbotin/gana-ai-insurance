"""
Operational Liability Claim State Manager for tracking claim intake progress.

Adapted for OperationalLiabilityClaim schema with provenance tracking.
"""

import uuid
from datetime import datetime
from typing import Any, Optional

from .schema import (
    OperationalLiabilityClaim,
    ClaimantInfo,
    IncidentInfo,
    OperationalImpactInfo,
    EvidenceChecklist,
    ConsistencyFlags,
    IncidentType,
    AssetType,
    ImpactSeverity,
    Provenance,
    SourceModality,
)


# Field definitions with priority, question text, and path
FIELD_DEFINITIONS = [
    # Priority 1: Claimant identification
    {
        "id": "claimant.name",
        "path": ["claimant", "name"],
        "priority": 1,
        "question": "What's your company name (policyholder)?",
        "required": True,
    },
    {
        "id": "claimant.policy_number",
        "path": ["claimant", "policy_number"],
        "priority": 1,
        "question": "And what's your policy number?",
        "required": True,
    },
    # Priority 2: Incident basics
    {
        "id": "incident.incident_type",
        "path": ["incident", "incident_type"],
        "priority": 2,
        "question": "What type of incident was it? You can describe it in your own words (e.g. pricing error, delay, misroute, loss, system outage).",
        "required": True,
    },
    {
        "id": "incident.incident_date",
        "path": ["incident", "incident_date"],
        "priority": 3,
        "question": "When did this incident occur?",
        "required": False,
    },
    # Priority 3: Location/System
    {
        "id": "incident.incident_location",
        "path": ["incident", "incident_location"],
        "priority": 3,
        "question": "Which system or route was affected?",
        "required": False,
    },
    # Priority 4: Description
    {
        "id": "incident.incident_description",
        "path": ["incident", "incident_description"],
        "priority": 4,
        "question": "Can you walk me through what happened?",
        "required": True,
    },
    # Priority 5: Operational impact details
    {
        "id": "operational_impact.asset_type",
        "path": ["operational_impact", "asset_type"],
        "priority": 5,
        "question": "What type of asset was affected (shipment, package, route, AI model, etc.)?",
        "required": False,
    },
    {
        "id": "operational_impact.system_component",
        "path": ["operational_impact", "system_component"],
        "priority": 5,
        "question": "Which system or component was involved?",
        "required": False,
    },
    # Priority 6: Cost and severity
    {
        "id": "operational_impact.impact_severity",
        "path": ["operational_impact", "impact_severity"],
        "priority": 6,
        "question": "How severe was the impact - minor, moderate, severe, or critical?",
        "required": False,
    },
    {
        "id": "operational_impact.estimated_liability_cost",
        "path": ["operational_impact", "estimated_liability_cost"],
        "priority": 6,
        "question": "Do you have an estimate of the cost or loss? If you have sold price and cost, I can use that.",
        "required": False,
    },
    # Priority 7: Contact info
    {
        "id": "claimant.contact_phone",
        "path": ["claimant", "contact_phone"],
        "priority": 7,
        "question": "What is the best phone number to reach you?",
        "required": False,
    },
    {
        "id": "claimant.contact_email",
        "path": ["claimant", "contact_email"],
        "priority": 7,
        "question": "What is your email address for claim updates?",
        "required": False,
    },
]


def _get_nested_value(obj: Any, path: list) -> Any:
    """Get a value from a nested path."""
    try:
        for key in path:
            if isinstance(key, int):
                if isinstance(obj, list) and len(obj) > key:
                    obj = obj[key]
                else:
                    return None
            else:
                obj = getattr(obj, key, None)
            if obj is None:
                return None
        return obj
    except (IndexError, AttributeError):
        return None


def _set_nested_value(obj: Any, path: list, value: Any) -> bool:
    """Set a value at a nested path. Returns True if successful."""
    try:
        for key in path[:-1]:
            if isinstance(key, int):
                if isinstance(obj, list):
                    while len(obj) <= key:
                        obj.append(None)
                    if obj[key] is None:
                        return False
                    obj = obj[key]
                else:
                    return False
            else:
                obj = getattr(obj, key, None)
                if obj is None:
                    return False

        final_key = path[-1]
        if isinstance(final_key, int):
            if isinstance(obj, list):
                while len(obj) <= final_key:
                    obj.append(None)
                obj[final_key] = value
            else:
                return False
        else:
            setattr(obj, final_key, value)
        return True
    except (IndexError, AttributeError, TypeError):
        return False


class OperationalClaimStateManager:
    """
    Manages the state of an operational liability claim during intake.

    Tracks which fields have been collected and determines the next
    question to ask based on priority and conditional logic.
    """

    def __init__(self, call_sid: Optional[str] = None, stream_sid: Optional[str] = None):
        """Initialize a new operational claim state manager."""
        self.claim = OperationalLiabilityClaim(
            claim_id=str(uuid.uuid4()),
            claimant=ClaimantInfo(),
            incident=IncidentInfo(),
            operational_impact=OperationalImpactInfo(),
            evidence=EvidenceChecklist(),
            consistency=ConsistencyFlags(),
            created_at=datetime.utcnow(),
        )

        # Store call metadata separately (not in schema)
        self.call_sid = call_sid
        self.stream_sid = stream_sid
        self.call_start_time = datetime.utcnow()

        # Conversation tracking
        self._conversation_turn = 0
        self._asked_fields: set[str] = set()
        self._transcript: list[dict] = []
        self._extraction_history: list[dict] = []

    def get_missing_fields(self, include_optional: bool = True) -> list[dict]:
        """
        Return list of fields not yet filled.

        Args:
            include_optional: If True, include optional fields; otherwise only required.

        Returns:
            List of field definitions that are missing values.
        """
        missing = []

        for field_def in FIELD_DEFINITIONS:
            # Skip optional fields if not requested
            if not include_optional and not field_def.get("required", False):
                continue

            # Check if field has value
            path = field_def["path"]
            value = _get_nested_value(self.claim, path)

            # Check for empty/default values
            if value is None or value == "":
                missing.append(field_def)
            elif isinstance(value, IncidentType) and value == IncidentType.UNKNOWN:
                missing.append(field_def)
            elif isinstance(value, AssetType) and value == AssetType.UNKNOWN:
                missing.append(field_def)
            elif isinstance(value, ImpactSeverity) and value == ImpactSeverity.UNKNOWN:
                missing.append(field_def)

        return missing

    def get_next_question(self) -> Optional[dict]:
        """
        Get the next question to ask based on priority.

        Returns:
            Field definition dict with 'question' key, or None if all done.
        """
        missing = self.get_missing_fields(include_optional=True)

        if not missing:
            return None

        # Sort by priority (lower is higher priority)
        missing.sort(key=lambda f: f["priority"])

        # Return the highest priority missing field
        return missing[0]

    def get_completion_percentage(self) -> float:
        """Calculate how complete the claim is (required fields only)."""
        required_fields = [f for f in FIELD_DEFINITIONS if f.get("required", False)]
        if not required_fields:
            return 100.0

        filled = 0

        for field_def in required_fields:
            path = field_def["path"]
            value = _get_nested_value(self.claim, path)

            if value is not None and value != "":
                # Check for non-default enum values
                if isinstance(value, IncidentType) and value != IncidentType.UNKNOWN:
                    filled += 1
                elif isinstance(value, AssetType) and value != AssetType.UNKNOWN:
                    filled += 1
                elif isinstance(value, ImpactSeverity) and value != ImpactSeverity.UNKNOWN:
                    filled += 1
                elif not isinstance(value, (IncidentType, AssetType, ImpactSeverity)):
                    filled += 1

        return (filled / len(required_fields)) * 100

    def is_complete(self) -> bool:
        """Check if all required fields have been collected."""
        missing_required = self.get_missing_fields(include_optional=False)
        return len(missing_required) == 0

    def apply_patch(self, patch: dict) -> list[str]:
        """
        Merge extracted data into current state.

        Args:
            patch: Dictionary with field paths as keys (dot notation) and values.

        Returns:
            List of field IDs that were updated.
        """
        updated = []

        for field_id, value in patch.items():
            if value is None:
                continue

            # Find the field definition
            field_def = next((f for f in FIELD_DEFINITIONS if f["id"] == field_id), None)

            if field_def:
                path = field_def["path"]
            else:
                # Try to parse the path from dot notation
                path = self._parse_path(field_id)

            # Convert enum values
            value = self._convert_enum_value(field_id, value)

            # Set the value
            if path and _set_nested_value(self.claim, path, value):
                updated.append(field_id)

                # Also set provenance if this is a provenance-tracked field
                self._set_provenance(field_id)

        # Record extraction
        if updated:
            self._extraction_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "turn": self._conversation_turn,
                "fields_updated": updated,
                "patch": patch,
            })

        return updated

    def _parse_path(self, field_id: str) -> list:
        """Parse a dot-notation field ID into a path list."""
        parts = field_id.split(".")
        path = []
        for part in parts:
            if part.isdigit():
                path.append(int(part))
            else:
                path.append(part)
        return path

    def _convert_enum_value(self, field_id: str, value: Any) -> Any:
        """Convert string values to appropriate enum types."""
        if not isinstance(value, str):
            return value

        value_lower = value.lower().strip()

        if field_id == "incident.incident_type":
            try:
                return IncidentType(value_lower)
            except ValueError:
                return IncidentType.UNKNOWN

        elif field_id == "operational_impact.asset_type":
            try:
                return AssetType(value_lower)
            except ValueError:
                return AssetType.UNKNOWN

        elif field_id == "operational_impact.impact_severity":
            try:
                return ImpactSeverity(value_lower)
            except ValueError:
                return ImpactSeverity.UNKNOWN

        return value

    def _set_provenance(self, field_id: str) -> None:
        """Set provenance for a field extracted from voice."""
        provenance = Provenance(
            source_modality=SourceModality.VOICE,
            confidence=0.8,  # Default confidence for voice extraction
            pointer=f"voice_turn:{self._conversation_turn}",
        )

        # Map field IDs to provenance fields
        provenance_map = {
            "incident.incident_date": ("incident", "incident_date_provenance"),
            "incident.incident_location": ("incident", "incident_location_provenance"),
            "incident.incident_description": ("incident", "incident_description_provenance"),
            "incident.incident_type": ("incident", "incident_type_provenance"),
            "operational_impact.asset_type": ("operational_impact", "asset_type_provenance"),
            "operational_impact.system_component": ("operational_impact", "system_component_provenance"),
            "operational_impact.estimated_liability_cost": ("operational_impact", "estimated_liability_cost_provenance"),
            "operational_impact.impact_severity": ("operational_impact", "impact_severity_provenance"),
        }

        if field_id in provenance_map:
            section, prov_field = provenance_map[field_id]
            section_obj = getattr(self.claim, section, None)
            if section_obj:
                setattr(section_obj, prov_field, provenance)

    def add_transcript_entry(self, role: str, content: str) -> None:
        """Add an entry to the conversation transcript."""
        self._transcript.append({
            "timestamp": datetime.utcnow().isoformat(),
            "turn": self._conversation_turn,
            "role": role,
            "content": content,
        })
        if role == "user":
            self._conversation_turn += 1

    def mark_field_asked(self, field_id: str) -> None:
        """Mark a field as having been asked about."""
        self._asked_fields.add(field_id)

    def was_field_asked(self, field_id: str) -> bool:
        """Check if a field has been asked about."""
        return field_id in self._asked_fields

    def to_dict(self) -> dict:
        """Export current claim state as dictionary."""
        data = self.claim.model_dump(mode="json")
        # Add call metadata
        data["_call_metadata"] = {
            "call_sid": self.call_sid,
            "stream_sid": self.stream_sid,
            "call_start_time": self.call_start_time.isoformat() if self.call_start_time else None,
        }
        data["_transcript"] = self._transcript
        data["_extraction_history"] = self._extraction_history
        return data

    def get_summary(self) -> str:
        """Generate a human-readable summary of collected information."""
        lines = []

        if self.claim.claimant.name:
            lines.append(f"Claimant: {self.claim.claimant.name}")
        if self.claim.claimant.policy_number:
            lines.append(f"Policy: {self.claim.claimant.policy_number}")
        if self.claim.incident.incident_type != IncidentType.UNKNOWN:
            lines.append(f"Incident Type: {self.claim.incident.incident_type.value}")
        if self.claim.incident.incident_date:
            lines.append(f"Date: {self.claim.incident.incident_date}")
        if self.claim.incident.incident_location:
            lines.append(f"Location: {self.claim.incident.incident_location}")
        if self.claim.incident.incident_description:
            lines.append(f"Description: {self.claim.incident.incident_description}")
        if self.claim.operational_impact.asset_type != AssetType.UNKNOWN:
            lines.append(f"Asset Type: {self.claim.operational_impact.asset_type.value}")
        if self.claim.operational_impact.system_component:
            lines.append(f"System: {self.claim.operational_impact.system_component}")
        if self.claim.operational_impact.impact_severity != ImpactSeverity.UNKNOWN:
            lines.append(f"Severity: {self.claim.operational_impact.impact_severity.value}")
        if self.claim.operational_impact.estimated_liability_cost:
            lines.append(f"Est. Cost: ${self.claim.operational_impact.estimated_liability_cost:,.2f}")

        completion = self.get_completion_percentage()
        lines.append(f"\nCompletion: {completion:.0f}%")

        return "\n".join(lines) if lines else "No information collected yet."

    def finalize(self) -> dict:
        """Finalize the claim and return the complete data."""
        # Update evidence checklist
        self._update_evidence_checklist()
        return self.to_dict()

    def _update_evidence_checklist(self) -> None:
        """Update the evidence checklist based on current state."""
        missing = []

        # Check what's missing
        if not self.claim.evidence.has_system_logs:
            missing.append("system_logs")
        if not self.claim.evidence.has_liability_assessment:
            missing.append("liability_assessment")
        if not self.claim.evidence.has_incident_report:
            missing.append("incident_report")

        self.claim.evidence.missing_evidence = missing


# Backwards compatibility aliases
PropertyClaimStateManager = OperationalClaimStateManager
FNOLStateManager = OperationalClaimStateManager
