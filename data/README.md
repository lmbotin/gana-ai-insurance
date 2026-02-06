# Data

This project uses **synthetic or publicly available** data only.

## Directory Structure

- `examples/` — Example claim JSON files (committed)
- `raw/` — Untouched inputs (ignored by git)
- `processed/` — Cleaned/standardized outputs (ignored by git)
- `synthetic/` — Generated sample claims for demos/evaluation (ignored by git)

## Example Files

The `examples/` folder contains sample `OperationalLiabilityClaim` JSON files:

| File | Description |
|------|-------------|
| `claim_complete.json` | Fully populated claim with all fields, high confidence scores, complete evidence |
| `claim_missing_evidence.json` | Claim with gaps - demonstrates `missing_evidence` list and low confidence fields |
| `claim_conflicting_evidence.json` | Claim with inconsistencies - demonstrates `ConsistencyFlags` with conflicts |

These examples can be used for:
- Testing schema validation
- Understanding the claim structure
- UI/demo development
- Integration testing
