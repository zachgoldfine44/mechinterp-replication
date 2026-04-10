# Anthropic Emotions Concepts Paper — local copy

> **This is a placeholder.** Replace this file with the full text of the
> paper being replicated. The pipeline reads it once at startup and
> exposes it via `PaperConfig.paper_text` to experiments and critique
> agents. Without this file populated, critique agents run blind and
> can't compare the harness's methodology to the paper's.

## How to populate this file

Pick one:

1. **From Anthropic's blog post** (HTML → markdown):
   ```bash
   curl -sL https://www.anthropic.com/research/<paper-slug> \
     | pandoc -f html -t markdown -o config/papers/emotions/paper.md
   ```

2. **From a PDF**:
   ```bash
   pdftotext -layout original.pdf - \
     | pandoc -f markdown -t markdown -o config/papers/emotions/paper.md
   ```

3. **Manually**: copy-paste the full text from the source. Keep section
   headings intact (`## Section`, `### Subsection`) so claims in
   `paper_config.yaml` can cite specific sections via the `paper_section`
   field.

## What this file is used for

- `src.core.config_loader.load_paper_config()` reads this file (path
  configurable via `paper.paper_text_path` in `paper_config.yaml`,
  defaults to `paper.md`).
- `src.core.critique.run_critique_pass()` includes the paper text in the
  prompts to the Claude and ChatGPT critic agents so they can verify
  methodology against the source.
- Future Claude sessions reading the harness use it as the **oracle**
  for what the paper actually says vs. what the harness implemented.

## Citation discipline

Once you've populated this file, every claim in `paper_config.yaml`
should add a `paper_section:` field, e.g.:

```yaml
- id: "claim_1"
  description: "Linear probes classify 12 emotions above chance"
  experiment_type: "probe_classification"
  paper_section: "Section 3.1, Figure 4"
  ...
```

The critique agents will use these section pointers to verify that the
methodology matches the paper.
