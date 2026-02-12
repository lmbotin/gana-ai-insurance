"""
SQLite-based claim storage.

Stores claims and processing results in a local SQLite database.
No external database setup required - just works.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

# Database file location
DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "claims.db"


@dataclass
class StoredClaim:
    """A claim record as stored in the database."""
    claim_id: str
    created_at: str
    updated_at: str
    status: str  # draft, submitted, processing, approved, denied, pending_review
    source: str  # text, image, voice, multimodal
    
    # Claim data (JSON)
    claimant: dict
    incident: dict
    property_damage: dict  # Legacy name, also stores operational_impact data
    evidence: dict
    
    # Processing results (JSON, nullable)
    validation_result: Optional[dict] = None
    fraud_result: Optional[dict] = None
    routing_result: Optional[dict] = None
    
    # Metadata
    call_sid: Optional[str] = None  # Twilio call ID if from voice
    session_id: Optional[str] = None
    notes: Optional[str] = None
    
    # Additional data (JSON, nullable)
    transcript: Optional[list] = None  # Conversation transcript
    consistency: Optional[dict] = None  # Consistency flags
    call_metadata: Optional[dict] = None  # Call timing, etc.
    operational_impact: Optional[dict] = None  # AI logistics claims


class ClaimStore:
    """
    SQLite-based storage for insurance claims.
    
    Usage:
        store = ClaimStore()
        
        # Save a claim
        claim_id = store.save(claim_data, source="text")
        
        # Retrieve
        claim = store.get(claim_id)
        
        # List all
        claims = store.list_all()
        
        # Update status
        store.update_status(claim_id, "approved")
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize the claim store."""
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Create tables if they don't exist."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS claims (
                    claim_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'draft',
                    source TEXT NOT NULL,
                    
                    -- Claim data (JSON)
                    claimant TEXT NOT NULL DEFAULT '{}',
                    incident TEXT NOT NULL DEFAULT '{}',
                    property_damage TEXT NOT NULL DEFAULT '{}',
                    evidence TEXT NOT NULL DEFAULT '{}',
                    
                    -- Processing results (JSON)
                    validation_result TEXT,
                    fraud_result TEXT,
                    routing_result TEXT,
                    
                    -- Metadata
                    call_sid TEXT,
                    session_id TEXT,
                    notes TEXT,
                    
                    -- Additional data (JSON)
                    transcript TEXT,
                    consistency TEXT,
                    call_metadata TEXT,
                    operational_impact TEXT
                )
            """)
            
            # Create indexes for common queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_claims_status ON claims(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_claims_created ON claims(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_claims_source ON claims(source)")
            
            # Add new columns if they don't exist (for existing databases)
            try:
                conn.execute("ALTER TABLE claims ADD COLUMN transcript TEXT")
            except sqlite3.OperationalError:
                pass  # Column already exists
            try:
                conn.execute("ALTER TABLE claims ADD COLUMN consistency TEXT")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE claims ADD COLUMN call_metadata TEXT")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE claims ADD COLUMN operational_impact TEXT")
            except sqlite3.OperationalError:
                pass

            # Resolved claims (policy number + amount paid; real-time or human-approved)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS resolved_claims (
                    claim_id TEXT PRIMARY KEY,
                    policy_number TEXT NOT NULL,
                    amount REAL NOT NULL,
                    resolved_at TEXT NOT NULL,
                    resolution_type TEXT NOT NULL,
                    resolved_by TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_resolved_policy ON resolved_claims(policy_number)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_resolved_at ON resolved_claims(resolved_at)")

            # Non-resolved claims (policy number, amount if any, reason not resolved)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS non_resolved_claims (
                    claim_id TEXT PRIMARY KEY,
                    policy_number TEXT NOT NULL,
                    amount REAL,
                    reason_not_resolved TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    notes TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_non_resolved_policy ON non_resolved_claims(policy_number)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_non_resolved_reason ON non_resolved_claims(reason_not_resolved)")
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _generate_claim_id(self) -> str:
        """Generate a unique claim ID."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        import random
        suffix = random.randint(1000, 9999)
        return f"CLM-{timestamp}-{suffix}"
    
    def save(
        self,
        claim_data: dict,
        source: str = "unknown",
        claim_id: Optional[str] = None,
        call_sid: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Save a claim to the database.
        
        Args:
            claim_data: Claim data dict (from PropertyDamageClaim.model_dump())
            source: Source of claim (text, image, voice, multimodal)
            claim_id: Optional custom claim ID
            call_sid: Twilio call SID if from voice
            session_id: Session ID for tracking
            
        Returns:
            The claim ID
        """
        claim_id = claim_id or claim_data.get("claim_id") or self._generate_claim_id()
        now = datetime.now().isoformat()
        
        # Extract transcript and metadata if present
        transcript = claim_data.get("_transcript", [])
        call_metadata = claim_data.get("_call_metadata", {})
        consistency = claim_data.get("consistency", {})
        
        with self._get_connection() as conn:
            # Support both property_damage and operational_impact (AI logistics) claims
            operational_impact = claim_data.get("operational_impact")
            conn.execute("""
                INSERT INTO claims (
                    claim_id, created_at, updated_at, status, source,
                    claimant, incident, property_damage, evidence,
                    call_sid, session_id,
                    transcript, consistency, call_metadata, operational_impact
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                claim_id,
                now,
                now,
                "submitted",
                source,
                json.dumps(claim_data.get("claimant", {})),
                json.dumps(claim_data.get("incident", {})),
                json.dumps(claim_data.get("property_damage", {})),
                json.dumps(claim_data.get("evidence", {})),
                call_sid,
                session_id,
                json.dumps(transcript) if transcript else None,
                json.dumps(consistency) if consistency else None,
                json.dumps(call_metadata) if call_metadata else None,
                json.dumps(operational_impact) if operational_impact is not None else None,
            ))
            conn.commit()
        
        return claim_id
    
    def save_from_pydantic(self, claim, source: str = "unknown", **kwargs) -> str:
        """
        Save a PropertyDamageClaim Pydantic model.
        
        Args:
            claim: PropertyDamageClaim instance
            source: Source of claim
            **kwargs: Additional metadata (call_sid, session_id)
            
        Returns:
            The claim ID
        """
        claim_data = claim.model_dump()
        return self.save(claim_data, source=source, **kwargs)
    
    def get(self, claim_id: str) -> Optional[StoredClaim]:
        """
        Retrieve a claim by ID.
        
        Returns:
            StoredClaim or None if not found
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM claims WHERE claim_id = ?",
                (claim_id,)
            ).fetchone()
            
            if row:
                return self._row_to_stored_claim(row)
        return None
    
    def list_all(
        self,
        status: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[StoredClaim]:
        """
        List claims with optional filtering.
        
        Args:
            status: Filter by status
            source: Filter by source
            limit: Max results
            offset: Pagination offset
            
        Returns:
            List of StoredClaim objects
        """
        query = "SELECT * FROM claims WHERE 1=1"
        params = []
        
        if status:
            query += " AND status = ?"
            params.append(status)
        
        if source:
            query += " AND source = ?"
            params.append(source)
        
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_stored_claim(row) for row in rows]
    
    def update_status(self, claim_id: str, status: str, notes: Optional[str] = None) -> bool:
        """
        Update claim status.
        
        Args:
            claim_id: Claim ID
            status: New status
            notes: Optional notes
            
        Returns:
            True if updated, False if claim not found
        """
        now = datetime.now().isoformat()
        
        with self._get_connection() as conn:
            if notes:
                result = conn.execute(
                    "UPDATE claims SET status = ?, updated_at = ?, notes = ? WHERE claim_id = ?",
                    (status, now, notes, claim_id)
                )
            else:
                result = conn.execute(
                    "UPDATE claims SET status = ?, updated_at = ? WHERE claim_id = ?",
                    (status, now, claim_id)
                )
            conn.commit()
            return result.rowcount > 0
    
    def save_processing_result(
        self,
        claim_id: str,
        validation_result: Optional[dict] = None,
        fraud_result: Optional[dict] = None,
        routing_result: Optional[dict] = None,
    ) -> bool:
        """
        Save processing results for a claim.
        
        Args:
            claim_id: Claim ID
            validation_result: Validation/completeness results
            fraud_result: Fraud analysis results
            routing_result: Routing decision results
            
        Returns:
            True if updated, False if claim not found
        """
        now = datetime.now().isoformat()
        
        updates = ["updated_at = ?"]
        params = [now]
        
        if validation_result is not None:
            updates.append("validation_result = ?")
            params.append(json.dumps(validation_result))
        
        if fraud_result is not None:
            updates.append("fraud_result = ?")
            params.append(json.dumps(fraud_result))
        
        if routing_result is not None:
            updates.append("routing_result = ?")
            params.append(json.dumps(routing_result))
        
        params.append(claim_id)
        
        with self._get_connection() as conn:
            result = conn.execute(
                f"UPDATE claims SET {', '.join(updates)} WHERE claim_id = ?",
                params
            )
            conn.commit()
            return result.rowcount > 0
    
    def save_resolved(
        self,
        claim_id: str,
        policy_number: str,
        amount: float,
        resolution_type: str = "real_time",
        resolved_by: Optional[str] = None,
    ) -> bool:
        """Save a resolved claim (real-time or human-approved). Idempotent on claim_id."""
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO resolved_claims
                (claim_id, policy_number, amount, resolved_at, resolution_type, resolved_by)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (claim_id, policy_number, amount, now, resolution_type, resolved_by or "system"))
            conn.commit()
        return True

    def save_non_resolved(
        self,
        claim_id: str,
        policy_number: str,
        reason_not_resolved: str,
        amount: Optional[float] = None,
        notes: Optional[str] = None,
    ) -> bool:
        """Save a non-resolved claim with reason. Idempotent on claim_id."""
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO non_resolved_claims
                (claim_id, policy_number, amount, reason_not_resolved, created_at, notes)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (claim_id, policy_number, amount, reason_not_resolved, now, notes))
            conn.commit()
        return True

    def delete(self, claim_id: str) -> bool:
        """Delete a claim."""
        with self._get_connection() as conn:
            result = conn.execute(
                "DELETE FROM claims WHERE claim_id = ?",
                (claim_id,)
            )
            conn.commit()
            return result.rowcount > 0
    
    def count(self, status: Optional[str] = None) -> int:
        """Count claims, optionally by status."""
        with self._get_connection() as conn:
            if status:
                row = conn.execute(
                    "SELECT COUNT(*) FROM claims WHERE status = ?",
                    (status,)
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) FROM claims").fetchone()
            return row[0]
    
    def _row_to_stored_claim(self, row: sqlite3.Row) -> StoredClaim:
        """Convert a database row to StoredClaim."""
        # Handle optional new columns that may not exist in older databases
        transcript = None
        consistency = None
        call_metadata = None
        
        try:
            if row["transcript"]:
                transcript = json.loads(row["transcript"])
        except (KeyError, IndexError):
            pass
        
        try:
            if row["consistency"]:
                consistency = json.loads(row["consistency"])
        except (KeyError, IndexError):
            pass
        
        try:
            if row["call_metadata"]:
                call_metadata = json.loads(row["call_metadata"])
        except (KeyError, IndexError):
            pass
        operational_impact = None
        try:
            if row["operational_impact"]:
                operational_impact = json.loads(row["operational_impact"])
        except (KeyError, IndexError, TypeError):
            pass

        return StoredClaim(
            claim_id=row["claim_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            status=row["status"],
            source=row["source"],
            claimant=json.loads(row["claimant"]),
            incident=json.loads(row["incident"]),
            property_damage=json.loads(row["property_damage"]),
            evidence=json.loads(row["evidence"]),
            validation_result=json.loads(row["validation_result"]) if row["validation_result"] else None,
            fraud_result=json.loads(row["fraud_result"]) if row["fraud_result"] else None,
            routing_result=json.loads(row["routing_result"]) if row["routing_result"] else None,
            call_sid=row["call_sid"],
            session_id=row["session_id"],
            notes=row["notes"],
            transcript=transcript,
            consistency=consistency,
            call_metadata=call_metadata,
            operational_impact=operational_impact,
        )


# =============================================================================
# Convenience Functions
# =============================================================================

@lru_cache
def get_claim_store() -> ClaimStore:
    """Get the default claim store (singleton)."""
    return ClaimStore()


def save_claim(claim_data: dict, source: str = "unknown", **kwargs) -> str:
    """Save a claim to the default store."""
    import logging
    logger = logging.getLogger(__name__)
    
    store = get_claim_store()
    logger.info(f"💾 Saving claim to database: {store.db_path.resolve()}")
    logger.info(f"   Source: {source}")
    logger.info(f"   Claimant: {claim_data.get('claimant', {})}")
    logger.info(f"   Incident: {claim_data.get('incident', {})}")
    
    claim_id = store.save(claim_data, source=source, **kwargs)
    logger.info(f"✅ Claim saved with ID: {claim_id}")
    return claim_id


def get_claim(claim_id: str) -> Optional[StoredClaim]:
    """Get a claim from the default store."""
    return get_claim_store().get(claim_id)


def list_claims(**kwargs) -> list[StoredClaim]:
    """List claims from the default store."""
    return get_claim_store().list_all(**kwargs)


def update_claim_status(claim_id: str, status: str, notes: Optional[str] = None) -> bool:
    """Update claim status in the default store."""
    return get_claim_store().update_status(claim_id, status, notes)
