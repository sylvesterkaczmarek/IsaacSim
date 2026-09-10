# Changelog

## [Unreleased]

### Changed
- Defer standalone backend selection until the first logging, admission-query, or configuration operation so hosts can
  attach their Carbonite interface after feature loggers have been constructed.

### Added
- Add C, C++, and Python logging facades over the pinned static Carbonite backend, including a stable C symbol baseline,
  immutable channel loggers, and source-location preservation.
- Add `{fmt}`-based C++ logging macros that skip argument evaluation for filtered records, thread-safe warning-once
  and deprecation-once macros, Python severity methods with opt-in per-call source capture and a `warn` alias, and
  backend-aware admission queries.
- Add C, C++, and Python `report` APIs for unconditional standard-output results, plus an explicit process-wide
  `flush()` operation. Prefix the unconditional report line with its logging channel.
- Add patch-style process-wide and per-channel Carbonite configuration for filtering, destinations, file output,
  rendering, asynchronous delivery, and multi-process settings.
- Add a standalone default logger that displays elapsed milliseconds without an absolute timestamp.
- Add an opt-in opaque host integration API for adapters that borrow an existing Carbonite backend.
