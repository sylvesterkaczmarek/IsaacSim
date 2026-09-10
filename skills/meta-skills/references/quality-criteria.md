# Quality Criteria — Skill Hierarchy Gates

Standing rubric for quality gates in skill hierarchies. Evolves via the feedback loop in the sibling `skill-hierarchy-authoring.md` note (link from parent `SKILL.md` only — do not chain reference hops).

## Structural (every skill)

| ID | Criterion | Severity | Origin |
|----|-----------|----------|--------|
| S1 | SKILL.md has valid YAML frontmatter (`name`, `description`) | error | baseline |
| S2 | Every script under `scripts/` is referenced from SKILL.md or a `references/*.md` | error | baseline |
| S3 | Routing table present when the skill has 3+ sub-skills | warning | baseline |
| S4 | Sub-skills link back to parent; no direct sub-skill-to-sub-skill calls | error | baseline |
| S5 | One level of nesting only (no `sub-skill/sub-sub-skill/`) | warning | baseline |

## Handoff contract (between routed stages)

| ID | Criterion | Severity | Origin |
|----|-----------|----------|--------|
| H1 | Each routing-table row defines a handoff contract | error | baseline |
| H2 | Output of stage N satisfies stage N+1 prerequisites before dispatch | error | baseline |
| H3 | Gate failure blocks the next stage (no silent skip) | error | baseline |

## Gate execution

| ID | Criterion | Severity | Origin |
|----|-----------|----------|--------|
| G1 | Gate has explicit pass/fail criteria (not subjective) | error | baseline |
| G2 | Fail action names the remediation (re-run, fix input, escalate) | error | baseline |
| G3 | Gate runs before output crosses a trust boundary (delivery, publish) | error | baseline |

## Distillation (post-delivery)

| ID | Criterion | Severity | Origin |
|----|-----------|----------|--------|
| D1 | Novel failure modes captured in the relevant sub-skill | warning | baseline |
| D2 | Standing rubric updated when a new gate check is needed | warning | baseline |
| D3 | Feedback logged with date, symptom, and action taken | warning | baseline |
