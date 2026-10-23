# Store all Job Postings; only Target-Role-tagged ones are searchable

All 123,849 Kaggle Job Postings are imported into SQLite once, and each is tagged with a Target Role (or none) from a reviewed title list; only tagged postings (~1.1k) are chunked and embedded into the ChromaDB Knowledge Base. This keeps embedding cheap and fast while making new Target Roles a quick re-tag plus embedding of the new postings, with no re-import, and lets market statistics cover the whole dataset.

## Consequences

- Statistics may cover all postings while search covers only tagged ones; answers must say which population they are based on.
