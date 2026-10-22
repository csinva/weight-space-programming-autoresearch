# Working agreements

- When creating Git commits or otherwise acting on GitHub for the user, use the user's name, Yash Mehta. Before committing, verify that the Git author name is `Yash Mehta`.
- Read `README.md` for the current RSA search and cross-dataset encoding contracts.
- During autonomous model search, only `model.py` and the proposer's notes/scratch are editable. Infrastructure maintenance is a separate task.
- LAION-fMRI and Triple-N supply RSA search feedback. NSD and TVSD are final-only encoding datasets; never feed their scores into search.
- Keep all subjects, fixed image policies, source filters, source metric kernels and explicit protocol provenance intact when maintaining `neuro_eval/`.
- Do not commit run databases, data files, caches, credentials or machine-local `.env` settings.
