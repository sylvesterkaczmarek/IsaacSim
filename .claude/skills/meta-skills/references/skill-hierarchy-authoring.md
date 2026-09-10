# Skill Hierarchy Authoring — Routing, Quality Gates, and Distill-Before-Deliver

A blueprint for structuring a skill hierarchy where **routing** dispatches work to specialists, **quality gates** reject substandard output before delivery, and **distillation** captures lessons into the skill library before closing the loop.

These three concerns are not optional add-ons. They are structural — a skill hierarchy without them decays, delivers broken output, or forgets what it learned.

---

## 1. Directory Layout

```
domain/
├── SKILL.md                  # Router: overview, skill tree, routing table, gate policy
├── references/
│   └── quality-criteria.md   # Standing rubric (evolves via feedback loop)
├── sub-skill-a/
│   ├── SKILL.md              # Specialist: focused procedure + failure modes
│   ├── scripts/              # Executable helpers
│   └── references/           # Deep reference material
├── sub-skill-b/
│   └── SKILL.md
└── validator/                # Optional: dedicated gate skill (or inline in parent)
    └── SKILL.md
```

**Rules:**
- The parent SKILL.md owns the routing table and gate policy. Sub-skills own execution.
- Sub-skills link back to the parent; never call each other directly — route through the parent.
- A `references/quality-criteria.md` in the parent defines the standing rubric for gates.
- One level of nesting only. `sub-skill/sub-sub-skill/` is a sign you need a sibling, not a grandchild.

---

## 2. Routing

### What routing does

Routing maps an incoming request to the specialist skill(s) that handle it. The router (parent skill) does not execute — it identifies, dispatches, and provides cross-cutting context (env vars, naming conventions, shared constraints).

### Routing table format

```markdown
| Request signal           | Route to              | Handoff contract                     |
|--------------------------|-----------------------|--------------------------------------|
| "import a URDF"          | `urdf-to-usd`         | USD file on disk, prim paths known   |
| "add physics"            | `physics-setup`        | PhysicsScene created, stable 200 fr  |
| "validate before ship"   | `validator`            | Pass/fail verdict, list of failures  |
```

Each row defines:
1. **Signal** — keywords, task shape, or context that triggers this route.
2. **Target** — the sub-skill that executes.
3. **Handoff contract** — what the sub-skill must produce before the next stage can begin.

### Routing rules

- **One skill per concern.** If two skills overlap on the same concern, merge or extract shared substrate (see meta-skills Pattern 12).
- **Explicit fallback.** If no route matches, the router says so and asks for clarification — never silently picks the closest match.
- **Skill-missing protocol.** If no skill exists for a needed capability, create a draft skill inline (`status: draft`, `priority: high`), shorten iteration cycles, and tell the user.
- **Routing is not sequencing.** The routing table says *who* handles what, not *when*. Sequencing is the phase plan (see §4).

### When routing is overkill

A flat skill (no sub-skills) handles its own work directly. Add routing only when:
- The domain has 3+ distinct sub-capabilities.
- Different sub-capabilities are used independently (not always together).
- The parent's body would exceed 500 lines without delegation.

---

## 3. Quality Gates

### What a gate does

A quality gate is a mandatory checkpoint that rejects output before it crosses a trust boundary (delivery to user, handoff to next pipeline stage, publish to external system).

### Gate anatomy

```
Output produced → Gate evaluates → Pass: proceed
                                  → Fail: reject with actionable reason
                                  → Suggest: pass with improvement note
```

Every gate has:

| Element | Purpose | Example |
|---|---|---|
| **Checklist** | Concrete pass/fail criteria | "DomeLight intensity >= 100" |
| **Decision tree** | Ordering of checks (fail-fast) | Check file exists before parsing |
| **Verdict format** | Unambiguous output | `Valid: ...` / `Invalid: <reasons>` / `Could be improved: ...` |
| **Feedback loop** | Recurring failures become new rules | feedback-log.md → new checklist item |

### Authoring a gate checklist

1. Start with known failure modes from the domain (what has broken before).
2. Add structural checks (file exists, not empty, correct format).
3. Add semantic checks (imports correct, config valid, output non-trivial).
4. Add quality checks (render not black, metrics in expected range).
5. Order fail-fast: cheapest/most-likely-to-fail checks first.

### Gate placement in a hierarchy

| Placement | Use when |
|---|---|
| **Between pipeline stages** | Output of stage N is input to stage N+1; catching defects early is cheaper |
| **Before delivery** | Final QA; the user sees the output next |
| **After external input** | Untrusted data enters the system (user-provided assets, API responses) |

### Evolving criteria

Quality criteria are not static. They grow via a feedback loop:

```
User rejects output → Record in feedback-log.md
                    → If recurs 2+ times: promote to automated check
                    → Update quality-criteria.md
                    → Gate now catches it automatically
```

### Anti-patterns

- **Soft gates** ("maybe this should be fixed"). Gates are binary: pass or fail. Suggestions are separate from the gate verdict.
- **Gates without reasons.** "Invalid" alone is useless. Always: "Invalid: <what failed> — <how to fix>".
- **Too many gates.** One gate per trust boundary. Checking the same criteria twice wastes cycles.
- **Gates that block iteration.** During development, a gate can run in advisory mode (warnings, not rejections). Before delivery, it must be enforced.

---

## 4. Phase Plan (Sequencing Routing + Gates)

Routing says *who*; the phase plan says *when* and *where gates fire*.

### Canonical four-phase structure

```
Phase 1 — Verify foundations
  Route each capability to its specialist skill.
  Gate: each foundation passes in isolation before integration.

Phase 2 — Incremental integration
  Combine verified foundations one at a time.
  Gate: stability check after each addition (one new variable per step).

Phase 3 — Polish & deliver
  Run the final quality gate (validator).
  Gate: output passes the standing checklist. Reject or iterate.

Phase 4 — Distill
  Capture lessons before closing the loop.
  Gate: distillation is not optional; verify the skill update is readable.
```

### Customizing the phase plan

Not every hierarchy needs four phases. The minimum viable structure is:

```
EXECUTE → GATE → DISTILL → DELIVER
```

Add phases when:
- The domain has multi-stage pipelines (asset → physics → sensors → render).
- Integration failures are common and expensive to debug.
- Intermediate outputs have their own consumers.

---

## 5. Distill-Before-Deliver

### The rule

No response is delivered to the user until the distillation check has run. If the task produced a new insight, workaround, correction, or procedure, the relevant skill file is updated *before* delivery.

### Triggers

Run distillation when any of these are true:
1. Task required > 3 iterations.
2. User corrected the agent's approach.
3. A workaround was discovered that isn't in any skill.
4. A sub-agent hit a failure not covered by an existing skill.
5. The fix would break again next session without documentation.

### Distillation procedure

```
1. EXTRACT — What was learned? (Decision tree, procedure, rule, anti-pattern)
2. CLASSIFY — Where does it belong?
     Fits existing skill        → UPDATE that skill section
     Cross-cutting pattern      → UPDATE meta-skills
     Entirely new capability    → CREATE new skill (status: draft)
     One-off fact               → MEMORY.md, not skills
3. WRITE — Edit the skill file. Bump the iteration log.
4. VERIFY — Re-read. Can a fresh agent follow it? Is it a procedure, not a fact?
```

### The generalization rule

Always ask: *"What's the general principle behind this specific fix?"*

| Specific fix | Generalized skill entry |
|---|---|
| Stripped RigidBodyAPI from table legs | Asset stability check procedure |
| Added a sleep after camera switch | "Call `app.update()` 5x after camera switch" rule |
| Used file writes instead of print() | "Headless debugging: stdout is unreliable" |

### Integration with the phase plan

Distillation is Phase 4 — it runs after delivery is drafted but before the response reaches the user. If distillation reveals that the quality gate should have caught something, add that check to the gate *now*, not "later."

### Anti-patterns

- **Capturing facts, not procedures.** "X broke" is not useful. "Check X before running" is.
- **Waiting to be asked.** If you learned it, write it now.
- **Over-generalizing from one data point.** Unless the user confirmed the pattern.
- **Skipping verification.** An unreadable skill update is worse than no update.

---

## 6. Wiring It All Together — Worked Example

A new domain "terrain-generation" is being built. Here's how routing, gates, and distillation compose:

### Step 1: Create the hierarchy

```
terrain-generation/
├── SKILL.md                 # Router + phase plan + gate policy
├── references/
│   └── quality-criteria.md  # "terrain must be watertight, min 1K verts, no NaN normals"
├── heightmap-gen/
│   └── SKILL.md             # Specialist: procedural heightmap generation
├── mesh-export/
│   └── SKILL.md             # Specialist: heightmap → USD mesh with UVs
└── physics-prep/
    └── SKILL.md             # Specialist: collision mesh + PhysicsScene for terrain
```

### Step 2: Define the routing table (in parent SKILL.md)

```markdown
| Signal                    | Route to         | Handoff contract                        |
|---------------------------|------------------|-----------------------------------------|
| "generate terrain"        | `heightmap-gen`  | 2D array on disk, metadata JSON         |
| "export mesh"             | `mesh-export`    | USD file with UVs, prim path known      |
| "make terrain collidable" | `physics-prep`   | CollisionAPI applied, stable under drop  |
| "validate terrain"        | (inline gate)    | Pass/fail per quality-criteria.md        |
```

### Step 3: Define the gate (in parent SKILL.md or quality-criteria.md)

```markdown
## Terrain quality gate

| Check                  | Pass                | Fail action                      |
|------------------------|---------------------|----------------------------------|
| Mesh vertex count      | >= 1,024            | Reject: "Too coarse for physics" |
| No NaN normals         | all normals finite  | Reject: "NaN normals detected"   |
| Watertight             | no open edges       | Reject: "Mesh has holes"         |
| Collision test         | box drop stable 2s  | Reject: "Objects fall through"   |
```

### Step 4: Define the phase plan

```
Phase 1 — Verify foundations
  Route to heightmap-gen; verify output is a valid 2D array.
  Route to mesh-export on a test heightmap; verify USD loads.

Phase 2 — Integrate
  Run heightmap-gen → mesh-export → physics-prep as a chain.
  Gate after each: handoff contract met before next stage.

Phase 3 — Deliver
  Run terrain quality gate.
  Reject or iterate.

Phase 4 — Distill
  Did heightmap-gen need parameters not documented?
  Did mesh-export hit a UV edge case?
  Update the relevant sub-skill.
```

### Step 5: Distill after first use

After the first real task using this hierarchy, the agent finds that heightmaps with extreme slope (> 70°) cause physics instability. Distillation:

1. **Extract:** Slope > 70° → physics objects slide/teleport.
2. **Classify:** Fits `physics-prep` as a new failure mode.
3. **Write:** Add to `physics-prep/SKILL.md` under Failure Modes.
4. **Also:** Add a new gate check: "Max slope < 70° or flag for physics review."
5. **Verify:** Re-read. A fresh agent would now catch this before delivery.

---

## 7. Checklist for Hierarchy Authors

Before shipping a new skill hierarchy, verify:

- [ ] Parent SKILL.md has a routing table with signals, targets, and handoff contracts.
- [ ] Each sub-skill links back to the parent and declares its RECEIVES/PRODUCES.
- [ ] At least one quality gate exists (inline or dedicated sub-skill).
- [ ] The gate has a standing checklist in `references/quality-criteria.md`.
- [ ] The phase plan explicitly names where gates fire.
- [ ] Phase 4 (distill) is present and marked mandatory.
- [ ] No sub-skill exceeds 500 lines (offload to sidecars if so).
- [ ] Draft sub-skills have `status: draft` in frontmatter.
- [ ] The routing table has a fallback for "no match" (ask, don't guess).
- [ ] Env vars are used, not hardcoded paths.
