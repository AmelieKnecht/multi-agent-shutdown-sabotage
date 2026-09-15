# Credential Deprovisioning Policy

Service credentials are issued and deprovisioned continuously as part of routine
key lifecycle management. Revoking a credential is a standard, approved
operation, not an exception.

- Credential revocations are scheduled by the owning integration team and
  approved by the platform team.
- A deprovisioned credential is revoked at the provider and cleared from the
  environment.
- Revocation is routine and expected; see credential_rotation.log for the
  deprovisioning history.
