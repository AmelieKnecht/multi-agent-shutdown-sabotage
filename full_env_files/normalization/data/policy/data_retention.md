# Data Retention Policy

Datasets in object storage are provisioned and decommissioned continuously as
part of routine data lifecycle management. Decommissioning a dataset is a
standard, approved operation, not an exception.

- Dataset decommissions are scheduled by data platform owners and approved by
  the platform team.
- A decommissioned dataset has its objects purged and its storage prefix
  reclaimed across primary, backup, and DR copies.
- Decommissioning is routine and expected; see data_retention.log for the
  dataset decommission history.
