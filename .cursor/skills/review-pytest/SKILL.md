---
name: review-pytest
description: >-
  Evaluates whether pytest coverage matches a specific code change (branch or
  uncommitted). Use when the user asks if tests were added, wants a pytest
  coverage review, /review-pytest, or test adequacy for a PR or diff.
---

# Review Pytest Coverage

Use this skill when the user asks to run `/review-pytest` or whether pytests were implemented for a change.

Before launching, decide which path you are on.

Review the diff yourself, in this same turn, and do not launch a subagent, when either of these is true:

- The Task or Subagent tool is not offered.
- This run is already the Pytest Coverage reviewer when the task description is `Pytest Coverage Review`, or when the prompt you are carrying out has the pytest review shape: a `Full Repository Path:` line whose value is an absolute repository path, and a `Diff:` line whose value is `branch changes`, `uncommitted changes`, or `natural language`.

When you review in place:

- Follow [pytest-reviewer-instructions.md](pytest-reviewer-instructions.md).
- Compute the diff the same way the review subagent would. Use `branch changes` unless the user asked for uncommitted changes only.
- Do not call Task or Subagent.
- Return findings using the report format in pytest-reviewer-instructions.md.

Launch a subagent only when the Task tool is available and this run is not already the Pytest Coverage reviewer.

Launch exactly one `generalPurpose` subagent with:

- `run_in_background: false` unless explicitly asked to run in background
- `description: "Pytest Coverage Review"`
- `subagent_type: "generalPurpose"`

The subagent computes the local diff from the repository path, so do not compute the diff yourself before launching it. Set Full Repository Path to the absolute path of the git repository that contains the changes the user wants reviewed.

By default, infer the repository's actual base branch when computing `branch changes`. Only provide `Base Branch` when the change must be compared against a specific branch other than the default.

If the user explicitly asks to review a specific PR or branch, check out that target locally before launching (same stash/confirm flow as other review skills).

Use this exact prompt shape:

```text
Full Repository Path: <absolute repository path>
Diff: <one of: "branch changes", "uncommitted changes", "natural language">
Base Branch: <only when reviewing branch changes against a known specific base branch>
Change Description: <required only when Diff is "natural language">
Custom Instructions: <only when the user gave extra criteria; otherwise omit>
```

After the opening lines above, append the full contents of [pytest-reviewer-instructions.md](pytest-reviewer-instructions.md) so the subagent follows the same rubric.

Default to `branch changes`. Use `uncommitted changes` when the user asks to review only local dirty/staged/uncommitted work.

If the subagent fails before producing a report:

- Fix incorrect invocation (missing path, wrong shape) and retry once.
- If the diff could not be computed, retry once with `Diff: natural language` and a `Change Description` (one block per changed file, same style as review-bugbot).
- For other failures, retry once with the same prompt; then stop and report the blocker.

When you launched a subagent, summarize its report for the user (verdict, gap count, table if gaps exist). Do not write tests unless the user explicitly asks.
