"""
Canonical claim schema for Track & Trace / AI Operational Liability claims.

Defines Pydantic models with provenance tracking for each extracted field.
Domain: Liability claims arising from AI-powered logistics and tracking system failures
(misroutes, delays, losses, prediction failures, data errors, etc.)
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, validator


# ============================================================================
# Enums (predefined list of allowed values)
# ============================================================================


class SourceModality(str, Enum):
    """Source of extracted information."""
    TEXT = "text"
    IMAGE = "image"
    DOCUMENT = "document"
    VOICE = "voice"
    LOG = "log"
    TELEMETRY = "telemetry"
    AUDIT_TRAIL = "audit_trail"


class IncidentType(str, Enum):
    """Type of operational incident causing liability."""
    MISROUTE = "misroute"           # Shipment sent to wrong destination
    DELAY = "delay"                 # Delivery delayed beyond SLA
    LOSS = "loss"                   # Shipment/package lost entirely
    DATA_ERROR = "data_error"       # Incorrect data entry or processing
    PREDICTION_FAILURE = "prediction_failure"  # AI model made incorrect prediction
    SYSTEM_OUTAGE = "system_outage" # System unavailability causing impact
    OTHER = "other"
    UNKNOWN = "unknown"


class AssetType(str, Enum):
    """Type of asset affected by the incident."""
    SHIPMENT = "shipment"       # Full shipment/consignment
    PACKAGE = "package"         # Individual package
    CONTAINER = "container"     # Shipping container
    AI_MODEL = "ai_model"       # AI/ML model or service
    SENSOR = "sensor"           # IoT sensor or tracking device
    ROUTE = "route"             # Delivery route or path
    PREDICTION = "prediction"   # AI-generated prediction/forecast
    DOCUMENT = "document"       # Shipping document, manifest, etc.
    OTHER = "other"
    UNKNOWN = "unknown"


class ImpactSeverity(str, Enum):
    """Severity assessment of operational impact."""
    MINOR = "minor"         # Limited impact, easily recoverable
    MODERATE = "moderate"   # Significant impact, requires intervention
    SEVERE = "severe"       # Major impact, substantial losses
    CRITICAL = "critical"   # Business-critical failure, urgent resolution needed
    UNKNOWN = "unknown"


# ============================================================================
# Provenance Model (this part checks where a specific part of the claim came from)
# ============================================================================


class Provenance(BaseModel):
    """
    Provenance metadata for an extracted field.

    Tracks where the information came from and confidence level.
    """
    source_modality: SourceModality
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    pointer: str = Field(description="Reference to source (e.g., 'text_span:0-50', 'image_id:img_001')")

    @validator('confidence')
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v


# ============================================================================
# Schema Sections
# ============================================================================


class ClaimantInfo(BaseModel):
    """Basic claimant information (minimal for Sprint 1)."""
    # all optional since at the start of the call you might not have all of this yet
    # (ai collects this during convo)
    name: Optional[str] = Field(None, description="Claimant full name")
    policy_number: Optional[str] = Field(None, description="Insurance policy number")
    contact_phone: Optional[str] = Field(None, description="Contact phone number")
    contact_email: Optional[str] = Field(None, description="Contact email address")


class IncidentInfo(BaseModel):
    """Information about the operational incident."""

    incident_date: Optional[datetime] = Field(None, description="When the incident occurred")
    incident_date_provenance: Optional[Provenance] = None

    incident_location: Optional[str] = Field(None, description="System/node/route where incident occurred (e.g., 'HUB-LAX-03', 'routing-engine-v2')")
    incident_location_provenance: Optional[Provenance] = None

    incident_description: Optional[str] = Field(None, description="Narrative description of what happened")
    incident_description_provenance: Optional[Provenance] = None

    incident_type: IncidentType = Field(default=IncidentType.UNKNOWN, description="Category of operational incident")
    incident_type_provenance: Optional[Provenance] = None


class OperationalImpactInfo(BaseModel):
    """Details about the affected asset and operational impact."""

    asset_type: AssetType = Field(default=AssetType.UNKNOWN, description="Type of asset affected")
    asset_type_provenance: Optional[Provenance] = None

    system_component: Optional[str] = Field(None, description="Specific system/subsystem affected (e.g., 'routing-engine', 'prediction-service')")
    system_component_provenance: Optional[Provenance] = None

    estimated_liability_cost: Optional[float] = Field(None, ge=0, description="Estimated financial liability (must be >= 0)")
    estimated_liability_cost_provenance: Optional[Provenance] = None

    impact_severity: ImpactSeverity = Field(default=ImpactSeverity.UNKNOWN, description="Severity of operational impact")
    impact_severity_provenance: Optional[Provenance] = None


class EvidenceChecklist(BaseModel):
    """Tracks what evidence has been provided."""

    has_system_logs: bool = Field(default=False, description="Whether system logs are provided")
    system_log_count: int = Field(default=0, ge=0, description="Number of system log files/entries")
    system_log_ids: List[str] = Field(default_factory=list, description="IDs/paths of system logs")

    has_liability_assessment: bool = Field(default=False, description="Whether liability/impact assessment is provided")

    has_incident_report: bool = Field(default=False, description="Whether incident report is provided")

    missing_evidence: List[str] = Field(
        default_factory=list,
        description="List of missing required evidence (e.g., 'liability_assessment', 'system_logs')"
    )


# this is some basic initial fraud detection
class ConsistencyFlags(BaseModel):
    """Flags for evidence consistency issues."""

    has_conflicts: bool = Field(default=False, description="Whether conflicts detected")
    conflict_details: List[str] = Field(
        default_factory=list,
        description="List of specific conflicts (e.g., 'date mismatch: text says Jan 1, image EXIF says Jan 5')"
    )

# ============================================================================
# Main Claim Schema
# ============================================================================


class OperationalLiabilityClaim(BaseModel):
    """
    Canonical schema for a Track & Trace / AI operational liability claim.

    Includes provenance tracking for all extracted fields.
    """

    # Unique identifier
    claim_id: str = Field(description="Unique claim identifier")

    # Core sections
    claimant: ClaimantInfo = Field(description="Claimant information")
    incident: IncidentInfo = Field(description="Incident details")
    operational_impact: OperationalImpactInfo = Field(description="Operational impact details")
    evidence: EvidenceChecklist = Field(description="Evidence completeness tracking")
    consistency: ConsistencyFlags = Field(
        default_factory=ConsistencyFlags,
        description="Consistency check results"
    )

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Claim creation timestamp")
    schema_version: str = Field(default="2.0.0", description="Schema version")

    class Config:
        """Pydantic config."""
        schema_extra = {
            "examples": [
                {
                    "claim_id": "CLM-2024-001",
                    "claimant": {
                        "name": "Acme Logistics Inc.",
                        "policy_number": "POL-TT-987654",
                        "contact_phone": "+1-555-0100",
                        "contact_email": "claims@acmelogistics.com"
                    },
                    "incident": {
                        "incident_date": "2024-01-15T14:30:00Z",
                        "incident_location": "HUB-LAX-03",
                        "incident_description": "AI routing model misclassified shipment priority, causing 48-hour delay for time-sensitive medical supplies",
                        "incident_type": "prediction_failure"
                    },
                    "operational_impact": {
                        "asset_type": "shipment",
                        "system_component": "routing-engine-v2",
                        "estimated_liability_cost": 15000.00,
                        "impact_severity": "severe"
                    },
                    "evidence": {
                        "has_system_logs": True,
                        "system_log_count": 3,
                        "system_log_ids": ["log_001.json", "log_002.json", "log_003.json"],
                        "has_liability_assessment": True,
                        "has_incident_report": False,
                        "missing_evidence": ["incident_report"]
                    }
                }
            ]
        }

    @validator('claim_id')
    def validate_claim_id(cls, v: str) -> str:
        """Ensure claim_id is not empty."""
        if not v or not v.strip():
            raise ValueError("claim_id cannot be empty")
        return v.strip()

    def to_json_schema(self) -> dict:
        """Export as JSON Schema."""
        return self.schema()

    def get_missing_evidence(self) -> List[str]:
        """Get list of missing required evidence."""
        return self.evidence.missing_evidence

    def has_consistency_issues(self) -> bool:
        """Check if there are any consistency issues."""
        return self.consistency.has_conflicts

    def get_consistency_issues(self) -> List[str]:
        """Get list of consistency issues."""
        return self.consistency.conflict_details
