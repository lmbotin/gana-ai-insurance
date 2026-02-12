"""
Operational liability claim processing workflow.

Handles post-call/chat claim processing:
- Validation & completeness check (delegates to fnol.checker)
- Fraud risk scoring
- Claim routing decisions

This runs AFTER the voice call or chat session completes.
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, TYPE_CHECKING

from openai import AsyncOpenAI

from ..fnol.checker import check_claim, CheckReport
from ..fnol.schema import OperationalLiabilityClaim
from ..storage.claim_store import get_claim_store
from ..policy import get_policy_service

if TYPE_CHECKING:
    from ..fnol.checker import CheckReport

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Data Classes
# =============================================================================


class ClaimPriority(str, Enum):
    """Claim priority levels."""
    URGENT = "urgent"      # Severe damage, ongoing issues
    HIGH = "high"          # Large losses, complex claims
    NORMAL = "normal"      # Standard claims
    LOW = "low"            # Minor damage, simple cases


class RoutingDecision(str, Enum):
    """Where to route the claim."""
    AUTO_APPROVE = "auto_approve"      # Straight-through processing
    STANDARD_QUEUE = "standard_queue"  # Normal adjuster queue
    SENIOR_ADJUSTER = "senior_adjuster"  # Complex/high-value
    SIU = "siu"                        # Special Investigation Unit (fraud)
    HUMAN_REVIEW = "human_review"      # Needs human decision


@dataclass
class ClaimProcessingResult:
    """Result of claim processing."""
    call_sid: str = ""
    
    # Validation
    is_complete: bool = False
    missing_fields: list[str] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)
    
    # Fraud analysis
    fraud_score: float = 0.0
    fraud_indicators: list[str] = field(default_factory=list)
    
    # Routing
    priority: ClaimPriority = ClaimPriority.NORMAL
    routing_decision: RoutingDecision = RoutingDecision.STANDARD_QUEUE
    routing_reason: str = ""
    
    # Output
    final_status: str = "pending"
    next_actions: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "call_sid": self.call_sid,
            "is_complete": self.is_complete,
            "missing_fields": self.missing_fields,
            "validation_errors": self.validation_errors,
            "fraud_score": self.fraud_score,
            "fraud_indicators": self.fraud_indicators,
            "priority": self.priority.value,
            "routing_decision": self.routing_decision.value,
            "routing_reason": self.routing_reason,
            "final_status": self.final_status,
            "next_actions": self.next_actions,
        }


# =============================================================================
# Helper Functions
# =============================================================================


def _apply_policy_step(claim_data: dict, result: "ClaimProcessingResult") -> None:
    """
    For AI logistics claims (with operational_impact): lookup policy, verify name,
    compute payout, and save to resolved_claims or non_resolved_claims.
    """
    claim_id = claim_data.get("claim_id") or _get_nested(claim_data, "claim_id")
    claimant = claim_data.get("claimant") or {}
    policy_number = claimant.get("policy_number") or _get_nested(claim_data, "claimant.policy_number")
    if not claim_id or not policy_number:
        return
    store = get_claim_store()
    svc = get_policy_service()
    policy = svc.get_policy(str(policy_number))
    if not policy:
        store.save_non_resolved(
            claim_id, str(policy_number), "policy_not_found", notes="Policy not found in database"
        )
        result.next_actions.append("Verify policy number with claimant; add policy if new.")
        return
    claimant_name = claimant.get("name") or _get_nested(claim_data, "claimant.name")
    name_ok = svc.verify_claimant_name(policy, claimant_name)
    if not name_ok:
        store.save_non_resolved(
            claim_id, str(policy_number), "name_mismatch",
            notes=f"Claimant name '{claimant_name}' does not match policy named_insured '{policy.named_insured}'"
        )
        result.next_actions.append("Confirm claimant name matches policy.")
        return
    # Coverage decision by LLM (name and policy number were already verified above, no LLM)
    try:
        coverage = svc.check_coverage_llm(claim_data, policy)
    except Exception as e:
        logger.warning("LLM coverage check failed, falling back to rule-based: %s", e)
        coverage = svc.check_coverage(claim_data, policy)
    if not coverage.is_covered:
        store.save_non_resolved(
            claim_id, str(policy_number), "incident_type_not_covered",
            notes=coverage.reason or "Claim not covered under policy"
        )
        result.next_actions.append(coverage.reason or "Claim not covered under policy; queue for review.")
        return
    # Payout: use LLM suggestion if available, else rule-based
    payout = coverage.suggested_payout_cap if coverage.suggested_payout_cap is not None else svc.compute_payout(claim_data, policy)
    fraud_flagged = result.fraud_score >= 0.6 or len(result.fraud_indicators) > 0
    can_auto = svc.can_auto_resolve(claim_data, policy, name_verified=True, fraud_flagged=fraud_flagged)
    if can_auto and payout is not None:
        store.save_resolved(claim_id, str(policy_number), float(payout), "real_time", resolved_by="system")
        result.final_status = "approved"
        result.routing_decision = RoutingDecision.AUTO_APPROVE
        result.routing_reason = f"Policy check passed; auto-approved payout ${payout:,.2f}"
        result.next_actions.append(f"Payout ${payout:,.2f} recorded; claim resolved.")
    else:
        reason = "needs_human_review"
        if payout is None:
            reason = "incident_type_not_covered"
        elif not can_auto and svc.get_required_extra_info(claim_data, policy):
            reason = "missing_evidence"
        store.save_non_resolved(
            claim_id, str(policy_number), reason, amount=float(payout) if payout is not None else None
        )
        result.next_actions.append("Claim queued for human review; policy check completed.")


def _get_nested(data: dict, path: str, default=None):
    """Get a nested value using dot notation."""
    keys = path.split(".")
    current: Any = data
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
        elif isinstance(current, list) and key.isdigit():
            idx = int(key)
            current = current[idx] if len(current) > idx else None
        else:
            return default
        if current is None:
            return default
    return current


# =============================================================================
# Processing Steps
# =============================================================================


def validate_claim_from_schema(claim: OperationalLiabilityClaim) -> tuple[bool, list[str], list[str], CheckReport]:
    """
    Validate an OperationalLiabilityClaim using the checker module.

    Args:
        claim: OperationalLiabilityClaim object

    Returns:
        Tuple of (is_complete, missing_fields, validation_errors, check_report)
    """
    # Delegate to the comprehensive checker
    report = check_claim(claim)

    # Additional data quality checks not covered by checker
    errors = []

    policy_number = claim.claimant.policy_number
    if policy_number and len(str(policy_number)) < 4:
        errors.append("Policy number appears invalid (too short)")

    phone = claim.claimant.contact_phone
    if phone and len(str(phone).replace("-", "").replace(" ", "")) < 10:
        errors.append("Phone number appears incomplete")

    # Check for estimated liability cost validity
    liability_cost = claim.operational_impact.estimated_liability_cost
    if liability_cost is not None and liability_cost < 0:
        errors.append("Estimated liability cost cannot be negative")

    # Add any contradictions from checker as errors
    errors.extend(report.contradictions)

    # Determine completeness based on checker score and required fields
    # Score >= 0.6 means tier 1 critical fields are mostly present
    is_complete = report.completeness_score >= 0.6 and len(errors) == 0

    return is_complete, report.missing_required_evidence, errors, report


def validate_claim(claim_data: dict) -> tuple[bool, list[str], list[str]]:
    """
    Validate claim completeness and data quality.
    Supports both property damage and operational (AI logistics / pricing) claims.
    For OperationalLiabilityClaim objects, use validate_claim_from_schema().

    Returns:
        Tuple of (is_complete, missing_fields, validation_errors)
    """
    missing = []
    errors = []
    has_operational = claim_data.get("operational_impact") is not None

    if has_operational:
        # Operational liability claims: name, policy, incident type, and liability cost or description
        required_fields = [
            ("claimant.name", "Claimant name"),
            ("claimant.policy_number", "Policy number"),
            ("incident.incident_type", "Incident type"),
        ]
        for path, label in required_fields:
            value = _get_nested(claim_data, path)
            if not value or value == "unknown":
                missing.append(label)
        cost = _get_nested(claim_data, "operational_impact.estimated_liability_cost")
        desc = _get_nested(claim_data, "incident.incident_description")
        if cost is None and not desc:
            missing.append("Estimated liability cost or incident description")
        if cost is not None:
            try:
                c = float(cost)
                if c < 0:
                    errors.append("Estimated liability cost cannot be negative")
            except (ValueError, TypeError):
                errors.append("Estimated liability cost is not a valid number")
    else:
        # Property damage claims
        required_fields = [
            ("claimant.name", "Claimant name"),
            ("claimant.policy_number", "Policy number"),
            ("incident.damage_type", "Type of damage"),
            ("incident.incident_description", "Incident description"),
        ]
        for path, label in required_fields:
            value = _get_nested(claim_data, path)
            if not value or value == "unknown":
                missing.append(label)
        repair_cost = _get_nested(claim_data, "property_damage.estimated_repair_cost")
        if repair_cost is not None:
            try:
                c = float(repair_cost)
                if c < 0:
                    errors.append("Estimated repair cost cannot be negative")
            except (ValueError, TypeError):
                errors.append("Estimated repair cost is not a valid number")

    # Common checks
    policy_number = _get_nested(claim_data, "claimant.policy_number")
    if policy_number and len(str(policy_number).strip()) < 3:
        errors.append("Policy number appears invalid (too short)")
    phone = _get_nested(claim_data, "claimant.contact_phone")
    if phone and len(str(phone).replace("-", "").replace(" ", "")) < 10:
        errors.append("Phone number appears incomplete")

    is_complete = len(missing) == 0 and len(errors) == 0
    return is_complete, missing, errors


async def analyze_fraud(claim_data: dict) -> tuple[float, list[str]]:
    """
    Analyze property damage claim for fraud indicators using LLM.
    
    Returns:
        Tuple of (fraud_score, fraud_indicators)
    """
    try:
        client = AsyncOpenAI()
        
        # Build analysis prompt
        system_prompt = """You are a fraud detection analyst for an insurance company.
Analyze the property damage claim data and identify potential fraud indicators.

Consider:
1. Timing anomalies (reported too late, vague dates)
2. Inconsistent details (damage type vs description mismatch)
3. Suspicious patterns (exaggerated damage, high repair estimates)
4. Red flags in the description (vague details, no witnesses)
5. Evidence gaps (no photos, no repair estimates)

Respond with JSON only:
{
    "fraud_score": 0.0-1.0 (higher = more suspicious),
    "indicators": ["list of specific concerns"],
    "reasoning": "brief explanation"
}

Be objective. Most claims are legitimate. Only flag genuine concerns."""

        damage_type = _get_nested(claim_data, 'incident.damage_type', 'unknown')
        description = _get_nested(claim_data, 'incident.incident_description', 'not provided')
        incident_date = _get_nested(claim_data, 'incident.incident_date', 'not provided')
        location = _get_nested(claim_data, 'incident.incident_location', 'not provided')
        property_type = _get_nested(claim_data, 'property_damage.property_type', 'unknown')
        severity = _get_nested(claim_data, 'property_damage.damage_severity', 'unknown')
        repair_cost = _get_nested(claim_data, 'property_damage.estimated_repair_cost', 'not provided')
        
        # Check evidence
        evidence = _get_nested(claim_data, 'evidence', {})
        has_photos = evidence.get('has_damage_photos', False) if isinstance(evidence, dict) else False
        has_estimate = evidence.get('has_repair_estimate', False) if isinstance(evidence, dict) else False

        user_prompt = f"""Analyze this property damage claim for fraud risk:

Damage Type: {damage_type}
Description: {description}
Date: {incident_date}
Location: {location}
Property Type: {property_type}
Severity: {severity}
Estimated Repair Cost: {repair_cost}
Has Damage Photos: {has_photos}
Has Repair Estimate: {has_estimate}
"""

        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=500,
        )
        
        result = json.loads(response.choices[0].message.content)
        
        return result.get("fraud_score", 0.0), result.get("indicators", [])
        
    except Exception as e:
        logger.error(f"Fraud analysis failed: {e}")
        # On error, return neutral score
        return 0.3, [f"Analysis error: {str(e)}"]


def determine_priority(claim_data: dict) -> ClaimPriority:
    """
    Determine claim priority from property damage or operational impact data.
    """
    # Operational (AI logistics / pricing) claims: use estimated_liability_cost
    liability_cost = _get_nested(claim_data, "operational_impact.estimated_liability_cost")
    if liability_cost is not None:
        try:
            cost = float(liability_cost)
            if cost > 20000:
                return ClaimPriority.URGENT
            if cost > 5000:
                return ClaimPriority.HIGH
            if cost > 1000:
                return ClaimPriority.NORMAL
            return ClaimPriority.LOW
        except (ValueError, TypeError):
            pass

    # Property damage claims
    severity = _get_nested(claim_data, "property_damage.damage_severity", "").lower()
    damage_type = _get_nested(claim_data, "incident.damage_type", "").lower()
    repair_cost = _get_nested(claim_data, "property_damage.estimated_repair_cost")

    if severity == "severe" or damage_type == "fire":
        return ClaimPriority.URGENT
    if repair_cost:
        try:
            cost = float(repair_cost)
            if cost > 10000:
                return ClaimPriority.HIGH
            elif cost < 1000:
                return ClaimPriority.LOW
        except (ValueError, TypeError):
            pass
    
    # Moderate damage = normal
    if severity == "moderate":
        return ClaimPriority.NORMAL
    
    # Minor damage = low
    if severity == "minor":
        return ClaimPriority.LOW
    
    return ClaimPriority.NORMAL


def route_claim(
    is_complete: bool,
    missing_fields: list[str],
    fraud_score: float,
    priority: ClaimPriority,
) -> tuple[RoutingDecision, str]:
    """
    Make routing decision based on all factors.
    
    Returns:
        Tuple of (routing_decision, routing_reason)
    """
    # High fraud score → SIU
    if fraud_score >= 0.7:
        return RoutingDecision.SIU, f"High fraud risk score ({fraud_score:.2f})"
    
    # Incomplete claim → Human review
    if not is_complete:
        return RoutingDecision.HUMAN_REVIEW, f"Missing required information: {', '.join(missing_fields)}"
    
    # Urgent priority → Senior adjuster
    if priority == ClaimPriority.URGENT:
        return RoutingDecision.SENIOR_ADJUSTER, "Urgent claim with severe damage"
    
    # Low priority, low fraud → Auto-approve eligible
    if priority == ClaimPriority.LOW and fraud_score < 0.2:
        return RoutingDecision.AUTO_APPROVE, "Low-risk, minor damage claim"
    
    # Default → Standard queue
    return RoutingDecision.STANDARD_QUEUE, "Standard processing"


def get_next_actions(routing_decision: RoutingDecision) -> tuple[str, list[str]]:
    """
    Determine final status and next actions based on routing.
    
    Returns:
        Tuple of (final_status, next_actions)
    """
    if routing_decision == RoutingDecision.AUTO_APPROVE:
        return "approved", [
            "Generate claim number",
            "Send confirmation to policyholder",
            "Schedule direct payment for minor repairs",
        ]
    
    elif routing_decision == RoutingDecision.SIU:
        return "under_investigation", [
            "Create SIU case file",
            "Flag for investigation",
            "Request additional documentation",
            "Hold all payments pending review",
        ]
    
    elif routing_decision == RoutingDecision.HUMAN_REVIEW:
        return "pending_review", [
            "Create review task",
            "Assign to available adjuster",
            "Request missing information from claimant",
        ]
    
    elif routing_decision == RoutingDecision.SENIOR_ADJUSTER:
        return "in_progress", [
            "Assign to senior adjuster",
            "Schedule property inspection",
            "Request contractor estimates",
            "Send acknowledgment to claimant",
        ]
    
    else:  # STANDARD_QUEUE
        return "in_progress", [
            "Assign to adjuster queue",
            "Send acknowledgment to policyholder",
            "Request damage photos if not provided",
            "Schedule follow-up call if needed",
        ]


# =============================================================================
# Main API
# =============================================================================


class ClaimProcessor:
    """
    Process claims through validation, fraud analysis, and routing.
    Supports operational liability claims (AI logistics) and legacy property damage claims.
    """

    async def process_claim(self, claim_data: dict, call_sid: str = "") -> ClaimProcessingResult:
        """
        Process a claim through the full workflow.

        Args:
            claim_data: The claim data from the voice call or chat session
            call_sid: Call/session SID for reference

        Returns:
            ClaimProcessingResult with all processing details
        """
        result = ClaimProcessingResult(call_sid=call_sid)
        
        # Step 1: Validate
        logger.info(f"Validating claim {call_sid}")
        result.is_complete, result.missing_fields, result.validation_errors = validate_claim(claim_data)
        
        # Step 2: Fraud analysis (skip if too incomplete)
        if len(result.missing_fields) <= 3:
            logger.info(f"Analyzing fraud risk for claim {call_sid}")
            result.fraud_score, result.fraud_indicators = await analyze_fraud(claim_data)
        else:
            logger.info("Skipping fraud analysis - claim too incomplete")
            result.fraud_score = 0.0
            result.fraud_indicators = []
        
        # Step 3: Determine priority
        logger.info(f"Determining priority for claim {call_sid}")
        result.priority = determine_priority(claim_data)
        
        # Step 4: Route
        logger.info(f"Routing claim {call_sid}")
        result.routing_decision, result.routing_reason = route_claim(
            result.is_complete,
            result.missing_fields,
            result.fraud_score,
            result.priority,
        )

        # Step 4b: Policy check (AI logistics operational liability)
        if claim_data.get("operational_impact") is not None:
            _apply_policy_step(claim_data, result)

        # Step 5: Determine next actions
        result.final_status, result.next_actions = get_next_actions(result.routing_decision)

        logger.info(f"Claim {call_sid} processed: {result.routing_decision.value} - {result.routing_reason}")
        
        return result


# Singleton instance
_processor: Optional[ClaimProcessor] = None


def get_claim_processor() -> ClaimProcessor:
    """Get or create the claim processor singleton."""
    global _processor
    if _processor is None:
        _processor = ClaimProcessor()
    return _processor


async def process_completed_call(claim_data: dict, call_sid: str = "") -> dict:
    """
    Convenience function to process a completed call.
    
    Call this after the voice call ends to run the claim through
    validation, fraud detection, and routing.
    
    Returns:
        Dictionary with processing results
    """
    processor = get_claim_processor()
    result = await processor.process_claim(claim_data, call_sid)
    return result.to_dict()
