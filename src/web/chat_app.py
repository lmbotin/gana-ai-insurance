"""
Web chat server for bidirectional claim intake.

Accepts text messages and photos, maintains conversation state,
and uses OpenAI Chat API for natural conversation.
"""

import asyncio
import base64
import json
import logging
import mimetypes
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from openai import AsyncOpenAI

from ..fnol.text_extractor import create_text_extractor
from ..fnol.config import ExtractionConfig
from ..policy import get_policy_service
from ..utils.config import settings

logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = AsyncOpenAI(api_key=settings.openai_api_key)

# In-memory session storage (in production, use Redis or DB)
chat_sessions: dict[str, dict] = {}

# Claim outcomes: successful vs failed (missing/wrong important data)
OUTCOMES_PATH = Path(__file__).resolve().parents[2] / "data" / "claim_outcomes.json"


def _record_claim_outcome(
    session_id: str,
    claim_id: Optional[str],
    status: str,
    reason: Optional[str] = None,
) -> None:
    """Append a claim outcome to data/claim_outcomes.json (successful vs failed)."""
    try:
        OUTCOMES_PATH.parent.mkdir(parents=True, exist_ok=True)
        if OUTCOMES_PATH.exists():
            with open(OUTCOMES_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {"successful": [], "failed": []}
        entry = {
            "session_id": session_id,
            "claim_id": claim_id,
            "status": status,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        if reason:
            entry["reason"] = reason
        if status == "success":
            data["successful"].append(entry)
        else:
            data["failed"].append(entry)
        with open(OUTCOMES_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.warning("Could not write claim outcome: %s", e)


# Operational claim field definitions — same flow as voice: what to ask, in what order.
# Voice uses get_missing_fields() + get_next_question(); chat uses this list for parity.
OPERATIONAL_FIELD_DEFINITIONS = [
    {"id": "claimant.name", "path": ["claimant", "name"], "priority": 1, "question": "What's your company name (policyholder)?", "required": True},
    {"id": "claimant.policy_number", "path": ["claimant", "policy_number"], "priority": 2, "question": "And what's your policy number?", "required": True},
    # Policy verification is done server-side; no field def, we gate before incident questions.
    {"id": "incident.incident_type", "path": ["incident", "incident_type"], "priority": 3, "question": "What type of incident was it? You can describe it in your own words (e.g. pricing error, delay, misroute, loss, system outage).", "required": True},
    {"id": "incident.incident_description", "path": ["incident", "incident_description"], "priority": 4, "question": "Can you walk me through what happened?", "required": True},
    {"id": "operational_impact.estimated_liability_cost", "path": ["operational_impact", "estimated_liability_cost"], "priority": 5, "question": "Do you have an estimate of the cost or loss? If you have sold price and cost, I can use that.", "required": False},
]


def _get_nested_claim_value(claim_data: dict, path: list) -> Optional[object]:
    """Get value from claim_data by path, e.g. ['claimant','name']."""
    cur = claim_data
    for key in path:
        cur = (cur or {}).get(key)
        if cur is None:
            return None
    if cur == "" or (isinstance(cur, str) and cur.strip().lower() == "unknown"):
        return None
    return cur


def _operational_missing_fields(claim_data: dict, policy_verified: bool) -> list[dict]:
    """Return field definitions that are still missing, in priority order. Same idea as voice state_manager.get_missing_fields()."""
    missing = []
    has_desc = _get_nested_claim_value(claim_data, ["incident", "incident_description"]) is not None
    has_cost = _get_nested_claim_value(claim_data, ["operational_impact", "estimated_liability_cost"]) is not None
    description_or_cost_ok = has_desc or has_cost
    for fd in OPERATIONAL_FIELD_DEFINITIONS:
        if fd["id"] == "incident.incident_description":
            # Satisfied if we have description OR estimated cost (same as voice: get key details)
            if description_or_cost_ok:
                continue
        val = _get_nested_claim_value(claim_data, fd["path"])
        if val is None:
            missing.append(fd)
        elif fd["id"] == "incident.incident_type" and isinstance(val, str) and val.strip().lower() == "unknown":
            missing.append(fd)
    missing.sort(key=lambda f: f["priority"])
    return missing


def _operational_next_question(missing: list[dict]) -> Optional[str]:
    """First missing field's question text. Same idea as voice state_manager.get_next_question()."""
    if not missing:
        return None
    return missing[0].get("question")


def _operational_completion_percentage(claim_data: dict, policy_verified: bool) -> float:
    """Required completion; 100 when we have name, policy, incident_type, and (description or cost)."""
    has_name = _get_nested_claim_value(claim_data, ["claimant", "name"]) is not None
    has_policy = _get_nested_claim_value(claim_data, ["claimant", "policy_number"]) is not None
    it = _get_nested_claim_value(claim_data, ["incident", "incident_type"])
    has_type = it is not None and (not isinstance(it, str) or it.strip().lower() != "unknown")
    has_desc = _get_nested_claim_value(claim_data, ["incident", "incident_description"]) is not None
    has_cost = _get_nested_claim_value(claim_data, ["operational_impact", "estimated_liability_cost"]) is not None
    description_or_cost_ok = has_desc or has_cost
    if has_name and has_policy and has_type and description_or_cost_ok:
        return 100.0
    required_count = 4  # name, policy, type, description_or_cost
    filled = sum([has_name, has_policy, has_type, 1 if description_or_cost_ok else 0])
    return (filled / required_count) * 100.0


# Wrap-up prompt when claim is complete — include a closing sentence, quick summary, and resolution status.
CLAIM_COMPLETE_CHAT_PROMPT = """You have collected the essential claim information. Give a proper closing that includes:

1. A quick summary of the claim in one or two sentences: company name, policy number, incident type, and key detail (e.g. estimated cost or loss).
2. A clear statement of whether the claim is resolved or not:
   - If the claim is fully resolved on this chat, say so (e.g. "Your claim has been resolved.").
   - If it is not resolved yet, say clearly: "Your claim has been recorded. It is not yet resolved — our team will review it and contact you within 1 to 2 business days to confirm next steps."
3. A closing sentence: ask if they need anything else, then say goodbye warmly (e.g. "Take care, we'll be in touch. Bye!").

Do not ask for more claim details or for company name or policy number again. Keep it short and human."""


def _build_chat_prompt(
    *,
    missing_fields: list[str],
    next_question: Optional[str],
    policy_issue: Optional[str],
    completion_pct: Optional[float] = None,
) -> str:
    """
    System prompt for the chat agent.

    Mirrors the voice-agent instruction style (policy-first, one question at a time),
    same flow: what to ask, what to check. Operational liability use case.
    """
    base = """You are Sarah, a friendly claims specialist at Gana Insurance in a live CHAT.

SOUND HUMAN (same as voice agent):
- Acknowledge what the user just said before asking the next thing (e.g. "Got it, thanks." or "Okay, so...").
- Never ask the same question twice if they already answered — use what they said and move on or clarify once.
- Use casual language and contractions: "I'll", "we'll", "that's", "don't".
- Ask ONE question at a time, then wait.

RULES:
- Do NOT invent or assume any information.
- Policy and name are already verified when you're in this flow; focus on the incident.

THIS USE CASE: Operational liability (AI logistics): e.g. pricing errors, delays, misroutes, losses, data errors, system outages.
- Collect: what happened (let them describe), and estimated cost or loss if they know it.
- If they mention sold price and cost (e.g. "sold for 3000, cost 10000"), the loss is cost minus sold price."""

    if missing_fields:
        base += f"\n\nFIELDS STILL NEEDED: {', '.join(missing_fields[:6])}"
    if next_question:
        base += f"\n\nSUGGESTED NEXT QUESTION (ask exactly one): {next_question}"
    if policy_issue:
        base += f"\n\nPOLICY CHECK: {policy_issue}"
    if completion_pct is not None:
        base += f"\n\nCLAIM STATUS: {completion_pct:.0f}% complete"

    return base


def _extract_claimant_from_text(text: str) -> dict:
    """Pull policyholder name and policy number from conversation (heuristic)."""
    out: dict = {}
    if not text or not text.strip():
        return out
    t = text.strip()
    # Policy number: use LAST match so corrections overwrite earlier value
    policy_patterns = [
        r"(?:policy\s*(?:number|numbe)?\s*[:\s]*|#\s*)(\d[\d\s\-]{3,})",
        r"\b(\d{5,8})\b",
    ]
    policy_m = None
    for pat in policy_patterns:
        for m in re.finditer(pat, t, re.IGNORECASE):
            policy_m = m
    if policy_m:
        num = re.sub(r"[\s\-]", "", policy_m.group(1))
        if len(num) >= 4:
            out["claimant.policy_number"] = num

    def _set_name(name: str) -> None:
        name = name.strip()
        if " and " in name:
            name = name.split(" and ")[0].strip()
        if 2 <= len(name) <= 120 and re.match(r"^[A-Za-z0-9\s&\'\-\.]+$", name):
            out["claimant.name"] = name

    # Name: "claim for X, policy" or "claim for X and policy" pattern (common in user messages)
    m = re.search(
        r"claim\s+for\s+([A-Za-z0-9\s&\'\-\.]+?)(?:\s*,\s*policy|\s+and\s+policy|\s+policy)",
        t,
        re.IGNORECASE,
    )
    if m:
        _set_name(m.group(1))
    # Name: explicit "I am X and policy number" / "X and policy number" first (most reliable)
    if "claimant.name" not in out:
        m = re.search(
            r"(?:i\s+am|we\s+are|company\s*(?:name)?\s*is|policyholder\s*is|this\s+is|(?:my\s+)?name\s+is)\s+([A-Za-z0-9\s&\'\-\.]+?)\s+and\s+(?:policy|the\s+policy)",
            t,
            re.IGNORECASE,
        )
        if m:
            _set_name(m.group(1))
    if "claimant.name" not in out:
        m = re.search(
            r"(?:i\s+am|my\s+name\s+is|i'?m|company\s*(?:name)?\s*is|policyholder\s*is|this\s+is|we\s+are)\s+([^.?!,:\n]+)",
            t,
            re.IGNORECASE,
        )
        if m:
            _set_name(m.group(1))
    if "claimant.name" not in out:
        m = re.search(r"(?:name\s+is|company)\s+([A-Za-z0-9\s&\'\-\.]+?)(?:\s+and\s+|\s*,\s*|\.|$)", t, re.IGNORECASE)
        if m:
            _set_name(m.group(1))
    # Standalone: if the whole message looks like a company name (no policy digits, 2–80 chars), use it as name
    if "claimant.name" not in out and not re.search(r"\d{4,}", t):
        candidate = t.strip()
        if 2 <= len(candidate) <= 80 and re.match(r"^[A-Za-z0-9\s&\'\-\.]+$", candidate):
            out["claimant.name"] = candidate
    return out


def _infer_pricing_error_loss(text: str) -> Optional[float]:
    """Heuristic: compute loss = cost - sold_price when both are stated."""
    if not text:
        return None
    t = text.lower()

    def _money_after(keyword_re: str) -> Optional[float]:
        m = re.search(keyword_re + r".{0,40}?(\$?\s*\d[\d,]*(?:\.\d+)?)", t)
        if not m:
            return None
        # Get the last group (the number), since keyword is in group 1
        raw = m.groups()[-1].replace("$", "").replace(",", "").strip()
        try:
            return float(raw)
        except Exception:
            return None

    sold = _money_after(r"(sold\s+for|sold\s+at|price\s+was|priced\s+at|negotiated\s+price\s+was|revenue\s+was|charged|we\s+charged|quoted\s+at|quoted)")
    cost = _money_after(r"(cost\s+was|cost\s+is|costing|our\s+cost\s+was|expense\s+was|cogs\s+was|cost\s+us|costs\s+us|actual\s+cost)")
    if sold is None or cost is None:
        return None
    loss = cost - sold
    return loss if loss >= 0 else None

app = FastAPI(
    title="Gana Insurance Chat",
    description="Bidirectional chatbot for operational liability claims",
    version="1.0.0",
)

# CORS for web frontend - configure via CORS_ALLOWED_ORIGINS env var
# Default: "*" (all origins) for development
# Production: Set CORS_ALLOWED_ORIGINS=https://your-frontend.com,https://other-domain.com
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Session timeout: after this many minutes of no activity, session is reset to avoid stale state
SESSION_TIMEOUT_MINUTES = 30


def get_or_create_session(session_id: str) -> dict:
    """Get or create a chat session. Resets session if it has timed out."""
    now = datetime.now(timezone.utc)
    if session_id not in chat_sessions:
        chat_sessions[session_id] = {
            "messages": [],
            "images": [],
            "claim_data": {},
            "extracted_fields": {},
            "policy_issue": None,
            "policy_verified": False,
            "last_activity": now.isoformat(),
        }
        return chat_sessions[session_id]
    session = chat_sessions[session_id]
    last = session.get("last_activity")
    if last:
        try:
            last_str = last if isinstance(last, str) else last.isoformat()
            last_dt = datetime.fromisoformat(last_str.replace("Z", "+00:00"))
        except Exception:
            last_dt = now
        if (now - last_dt).total_seconds() > SESSION_TIMEOUT_MINUTES * 60:
            # Timeout: reset session so user starts fresh and we don't use stale/locked state
            session["messages"] = []
            session["images"] = []
            session["extracted_fields"] = {}
            session["policy_issue"] = None
            session["policy_verified"] = False
    session["last_activity"] = now.isoformat()
    return session


@app.post("/api/chat")
async def chat_endpoint(
    message: str = Form(...),
    session_id: Optional[str] = Form(None),
    images: List[UploadFile] = File(default=[]),
):
    """
    Chat endpoint: accepts text message and optional images.
    
    Returns:
        {
            "session_id": "...",
            "assistant_message": "...",
            "claim_data": {...},  # Extracted claim info so far
            "is_complete": bool,
            "next_question": "..."  # Suggested next question
        }
    """
    # Get or create session
    sid = session_id or str(uuid.uuid4())
    session = get_or_create_session(sid)
    
    # Process uploaded images (encode as base64 for OpenAI vision API)
    image_contents = []
    image_paths = []
    if images:
        temp_dir = Path(tempfile.gettempdir()) / "gana_chat_images"
        temp_dir.mkdir(exist_ok=True)
        for img in images:
            if img.filename:
                content = await img.read()
                # Save to temp file
                img_path = temp_dir / f"{sid}_{img.filename}"
                with open(img_path, "wb") as f:
                    f.write(content)
                image_paths.append(str(img_path))
                session["images"].append(str(img_path))
                
                # Encode as base64 for OpenAI
                mime_type, _ = mimetypes.guess_type(img.filename)
                if not mime_type:
                    mime_type = "image/jpeg"  # Default
                image_contents.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64.b64encode(content).decode('utf-8')}"
                    }
                })
                logger.info(f"Processed image: {img_path} ({len(content)} bytes)")
    
    # Build user message content (text + images)
    user_content = [{"type": "text", "text": message}]
    user_content.extend(image_contents)
    
    # Add user message to conversation
    user_message = {"role": "user", "content": user_content}
    session["messages"].append(user_message)

    # ----------------------------
    # Extract fields + policy check
    # ----------------------------
    try:
        # Combine all user text messages so far
        all_user_text_parts = []
        for msg in session["messages"]:
            if msg["role"] == "user":
                if isinstance(msg["content"], str):
                    all_user_text_parts.append(msg["content"])
                elif isinstance(msg["content"], list):
                    # Extract text from content array
                    for item in msg["content"]:
                        if item.get("type") == "text":
                            all_user_text_parts.append(item.get("text", ""))
        all_user_text = " ".join(all_user_text_parts)
        current_message_text = message.strip() if isinstance(message, str) else ""

        # Claimant: extract from conversation; once policy was verified, never overwrite (stops wrong name from earlier in chat replacing correct one)
        claimant_locked = session.get("policy_verified") is True
        claimant_from_all = _extract_claimant_from_text(all_user_text)
        claimant_from_latest = _extract_claimant_from_text(current_message_text) if current_message_text else {}
        if not claimant_locked:
            for key in ("claimant.name", "claimant.policy_number"):
                if claimant_from_latest.get(key):
                    session["extracted_fields"][key] = claimant_from_latest[key]
                elif claimant_from_all.get(key):
                    session["extracted_fields"][key] = claimant_from_all[key]

        # Only run heavy LLM extract when policy is (or might be) verified — reduces latency until then
        has_name = bool(session["extracted_fields"].get("claimant.name"))
        has_policy = bool(session["extracted_fields"].get("claimant.policy_number"))
        policy_ok_so_far = has_name and has_policy
        if policy_ok_so_far:
            try:
                svc = get_policy_service()
                policy = svc.get_policy(str(session["extracted_fields"].get("claimant.policy_number", "")))
                if policy and svc.verify_claimant_name(policy, str(session["extracted_fields"].get("claimant.name", ""))):
                    policy_ok_so_far = True  # will run full extract
                else:
                    policy_ok_so_far = False
            except Exception:
                policy_ok_so_far = False
        run_full_extract = policy_ok_so_far

        if run_full_extract:
            cfg = ExtractionConfig(
                llm_provider="openai",
                llm_model=settings.openai_extraction_model,
                api_key=settings.openai_api_key,
            )
            extractor = create_text_extractor(cfg)
            extracted = extractor.extract(all_user_text)
            for key, value in claimant_from_all.items():
                if value:
                    extracted[key] = value
            for key, value in claimant_from_latest.items():
                if value:
                    extracted[key] = value
            try:
                it = (extracted.get("incident_type") or "").lower()
                if it == "pricing_error" and extracted.get("estimated_liability_cost") is None:
                    inferred = _infer_pricing_error_loss(all_user_text)
                    if inferred is not None:
                        extracted["estimated_liability_cost"] = inferred
            except Exception:
                pass
            # When policy already verified, never overwrite stored name/policy (extracted may contain wrong name from earlier in chat)
            locked = session.get("policy_verified") is True
            for key, value in extracted.items():
                if value is None or value == "unknown":
                    continue
                if locked and key in ("claimant.name", "claimant.policy_number"):
                    continue
                session["extracted_fields"][key] = value
        else:
            # Keep only claimant fields updated (already set above); incident fields stay from previous turns
            pass
        
        # Build claim_data dict for response
        claim_data = {
            "claimant": {
                "name": session["extracted_fields"].get("claimant.name"),
                "policy_number": session["extracted_fields"].get("claimant.policy_number"),
            },
            "incident": {
                "incident_type": session["extracted_fields"].get("incident_type"),
                "incident_description": session["extracted_fields"].get("incident_description"),
            },
            "operational_impact": {
                "estimated_liability_cost": session["extracted_fields"].get("estimated_liability_cost"),
            },
        }
        
        # Policy check (non-LLM): verify policy exists and name matches — same as voice agent
        policy_issue = None
        policy_named_insured: Optional[str] = None
        has_name = bool(claim_data["claimant"]["name"])
        has_policy = bool(claim_data["claimant"]["policy_number"])
        if has_name and has_policy:
            try:
                svc = get_policy_service()
                policy = svc.get_policy(str(claim_data["claimant"]["policy_number"]))
                if not policy:
                    policy_issue = "policy_not_found"
                elif not svc.verify_claimant_name(policy, str(claim_data["claimant"]["name"])):
                    policy_issue = "name_mismatch"
                    policy_named_insured = getattr(policy, "named_insured", None) or "our records"
            except Exception as e:
                logger.debug(f"Policy check skipped: {e}")
        session["policy_issue"] = policy_issue
        session["policy_named_insured"] = policy_named_insured
        policy_verified = has_name and has_policy and not policy_issue
        if policy_verified:
            session["policy_verified"] = True  # Lock claimant so later extraction cannot overwrite with wrong name/policy

        # Same flow as voice: missing fields + next question from field definitions
        missing_defs = _operational_missing_fields(claim_data, policy_verified)
        if not has_name:
            missing_fields = ["company name"]
            next_question = "To get started I need the policyholder name (company or your name) and policy number. What name should I use?"
        elif not has_policy:
            missing_fields = ["policy number"]
            next_question = "And what's your policy number?"
        elif policy_issue:
            # Company name and policy number must match the database — do not proceed until they do
            missing_fields = ["policy verification"]
            next_question = "recheck"
        else:
            # Only when name and policy match our records do we ask about the incident
            missing_fields = [f["id"] for f in missing_defs]
            next_question = _operational_next_question(missing_defs)

        completion_pct = _operational_completion_percentage(claim_data, policy_verified)
        is_complete = policy_verified and len(missing_defs) == 0

    except Exception as e:
        logger.warning(f"Extraction error (non-fatal): {e}")
        claim_data = {}
        is_complete = False
        next_question = "What's your company name (policyholder) and policy number?"
        missing_fields = ["company name", "policy number"]
        session["policy_issue"] = None

    # ----------------------------
    # Decide assistant response
    # ----------------------------
    # Gate until we have name, policy, AND they match the database; only then ask about the incident
    must_gate = "company name" in missing_fields or "policy number" in missing_fields or "policy verification" in missing_fields

    if must_gate:
        name = (claim_data.get("claimant") or {}).get("name") or ""
        policy_num = (claim_data.get("claimant") or {}).get("policy_number") or ""
        p_issue = session.get("policy_issue")
        if not name and not policy_num:
            assistant_message = "To get started I need your company name (policyholder) and policy number so we can check they match our records. What name should I use?"
        elif not policy_num:
            assistant_message = "And what's your policy number? I'll check it and your company name together."
        elif not name:
            assistant_message = "What's your company name (policyholder)?"
        elif p_issue == "policy_not_found":
            assistant_message = (
                "Thanks for that. The policy number you gave isn't in our system. "
                "Please double-check the policy number — we can only continue when it matches our records."
            )
        elif p_issue == "name_mismatch":
            assistant_message = (
                "The company name and policy number don't match what we have on file. "
                "Please double-check both — we can only continue when they match our records."
            )
        else:
            assistant_message = "Please double-check the company name and policy number so they match our records."
    else:
        # Same as voice: wrap-up prompt when complete, otherwise guided by next question + status
        if is_complete:
            system_prompt = CLAIM_COMPLETE_CHAT_PROMPT
        else:
            system_prompt = _build_chat_prompt(
                missing_fields=missing_fields,
                next_question=next_question,
                policy_issue=session.get("policy_issue"),
                completion_pct=completion_pct,
            )

        # Build conversation history for OpenAI (last 10 messages)
        conversation = [{"role": "system", "content": system_prompt}]
        for msg in session["messages"][-10:]:
            if isinstance(msg["content"], str):
                conversation.append({"role": msg["role"], "content": msg["content"]})
            else:
                conversation.append(msg)

        # Get assistant response from OpenAI
        try:
            response = await client.chat.completions.create(
                model="gpt-4o",  # GPT-4o supports vision
                messages=conversation,
                temperature=0.6,
            )
            assistant_message = response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI chat error: {e}")
            raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

    session["messages"].append({"role": "assistant", "content": assistant_message})

    # Slight delay so the response feels less instant (optional “typing” feel)
    await asyncio.sleep(0.5)
    return JSONResponse({
        "session_id": sid,
        "assistant_message": assistant_message,
        "claim_data": claim_data,
        "is_complete": is_complete,
        "next_question": next_question,
        "has_images": len(image_paths) > 0,
        "policy_issue": session.get("policy_issue"),
    })


@app.get("/api/chat/session/{session_id}")
async def get_session(session_id: str):
    """Get current session state."""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = chat_sessions[session_id]
    return JSONResponse({
        "session_id": session_id,
        "messages": session["messages"],
        "claim_data": session.get("claim_data", {}),
        "extracted_fields": session.get("extracted_fields", {}),
    })


@app.post("/api/chat/submit")
async def submit_claim(session_id: str = Form(...)):
    """
    Submit the claim for processing (after conversation is complete).
    
    Runs the claim through validation, policy check, and routing.
    """
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = chat_sessions[session_id]
    
    # Build full claim from conversation
    all_user_text_parts = []
    for msg in session["messages"]:
        if msg["role"] == "user":
            if isinstance(msg["content"], str):
                all_user_text_parts.append(msg["content"])
            elif isinstance(msg["content"], list):
                # Extract text from content array
                for item in msg["content"]:
                    if item.get("type") == "text":
                        all_user_text_parts.append(item.get("text", ""))
    all_user_text = " ".join(all_user_text_parts)
    
    # Build claim_data dict matching OperationalLiabilityClaim schema
    try:
        # Extract estimated_liability_cost (may be None)
        estimated_cost = session["extracted_fields"].get("estimated_liability_cost")
        if estimated_cost is not None:
            try:
                estimated_cost = float(estimated_cost)
            except (ValueError, TypeError):
                estimated_cost = None
        
        claim_data = {
            "claim_id": f"CLM-{uuid.uuid4().hex[:8].upper()}",
            "claimant": {
                "name": session["extracted_fields"].get("claimant.name") or "",
                "policy_number": session["extracted_fields"].get("claimant.policy_number") or "",
                "contact_phone": session["extracted_fields"].get("claimant.contact_phone"),
                "contact_email": session["extracted_fields"].get("claimant.contact_email"),
            },
            "incident": {
                "incident_date": session["extracted_fields"].get("incident_date"),
                "incident_location": session["extracted_fields"].get("incident_location"),
                "incident_description": session["extracted_fields"].get("incident_description"),
                "incident_type": session["extracted_fields"].get("incident_type") or "unknown",
            },
            "operational_impact": {
                "asset_type": session["extracted_fields"].get("asset_type") or "unknown",
                "system_component": session["extracted_fields"].get("system_component"),
                "estimated_liability_cost": estimated_cost,
                "impact_severity": session["extracted_fields"].get("impact_severity") or "unknown",
            },
            "evidence": {
                "has_system_logs": False,
                "has_liability_assessment": estimated_cost is not None,
                "has_incident_report": len(session.get("images", [])) > 0,
            },
        }
        
        # Process through routing workflow
        from ..routing import get_claim_processor
        processor = get_claim_processor()
        result = await processor.process_claim(claim_data, call_sid=f"chat-{session_id}")
        _record_claim_outcome(session_id, claim_data["claim_id"], "success")

        # Build a short summary and resolution message for the user
        claimant = claim_data.get("claimant") or {}
        incident = claim_data.get("incident") or {}
        impact = claim_data.get("operational_impact") or {}
        parts = [
            f"Company: {claimant.get('name') or '—'}",
            f"Policy: {claimant.get('policy_number') or '—'}",
            f"Incident: {incident.get('incident_type') or '—'}",
        ]
        if impact.get("estimated_liability_cost") is not None:
            parts.append(f"Estimated cost: ${float(impact['estimated_liability_cost']):,.2f}")
        summary = ". ".join(parts)
        resolved = getattr(result, "final_status", None) == "approved"
        if resolved:
            resolution_msg = "Your claim has been processed and is resolved."
        else:
            resolution_msg = "Your claim has been recorded and is not yet resolved — our team will review it and contact you within 1 to 2 business days."
        message = f"Summary of the claim we just processed: {summary}. {resolution_msg}"

        return JSONResponse({
            "success": True,
            "claim_id": claim_data["claim_id"],
            "result": result.to_dict(),
            "summary": summary,
            "resolved": resolved,
            "message": message,
        })

    except Exception as e:
        logger.error(f"Claim submission error: {e}", exc_info=True)
        _record_claim_outcome(session_id, None, "failed", reason=str(e))
        raise HTTPException(status_code=500, detail=f"Submission error: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint - serve chat interface."""
    chat_html_path = Path(__file__).parent.parent.parent / "static" / "chat.html"
    if chat_html_path.exists():
        return FileResponse(chat_html_path)
    return {
        "service": "Gana Insurance Chat",
        "status": "running",
        "endpoints": {
            "GET /": "Chat interface (HTML)",
            "POST /api/chat": "Send a message (with optional images)",
            "GET /api/chat/session/{session_id}": "Get session state",
            "POST /api/chat/submit": "Submit claim for processing",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.web.chat_app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
