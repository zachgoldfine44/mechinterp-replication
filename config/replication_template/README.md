# Replication template

Copy this whole directory into
`config/papers/{paper_id}/replications/{replication_id}/` to scaffold a
new replication.

```bash
cp -r config/replication_template \
  config/papers/{paper_id}/replications/{replication_id}
```

Then edit:

1. **`paper_config.yaml`** — fill in paper metadata, your chosen claims,
   thresholds, and experiment params. Set `replication.id`, `replication.replicator`,
   `replication.github_handle`, `replication.date`, and `replication.models_tested`
   to match your run.
2. **`stimuli_config.yaml`** — define your stimulus sets. Either point
   at an existing HuggingFace dataset or hand-write JSON files under
   `stimuli/`.
3. **`metadata.yaml`** — one-line summary for the README table. Regenerate
   the table after committing: `python scripts/generate_replications_table.py`.

**Do not** edit `config/papers/{paper_id}/paper.md` unless you are the
first person to replicate this paper. That file is the shared oracle
across all replications.

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for the full submission flow.
