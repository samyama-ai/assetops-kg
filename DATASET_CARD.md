---
license: other
pretty_name: AssetOps Knowledge Graph
tags:
  - knowledge-graph
  - samyama
  - property-graph
  - industrial
language:
  - en
size_categories:
  - 10K<n<100K
---

# Dataset Card for `assetops-kg`

**12,647 nodes. 12,662 edges. IBM AssetOpsBench at 99% accuracy -- deterministic graph queries, zero LLM tokens.**

> Part of the **Samyama** ecosystem. This card describes the dataset; the repository
> holds the loader and source-data specifics.

## Structure

**9 node labels** -- Equipment, Sensor, FailureMode, WorkOrder, Location, Site, Event, AnomalyEvent, AlertEvent

**5 edge types** -- CONTAINS_LOCATION, CONTAINS_EQUIPMENT, HAS_SENSOR, FOR_EQUIPMENT, MONITORS

**Data source** -- [IBM AssetOpsBench](https://github.com/IBM/AssetOpsBench) (139 scenarios, 9 asset classes)

## Provenance and licence

Apache 2.0

> ⚠️ **The licence above covers this repository's code, not the data.** This graph is
> derived from an upstream source ([IBM AssetOpsBench](https://github.com/IBM/AssetOpsBench) (139 scenarios, 9 asset classes)), whose
> own terms govern redistribution and are **not stated here**. Establish and record them
> before redistributing or quoting this dataset. The frontmatter is therefore
> `license: other` rather than `apache-2.0`.

## Reproducing

The loader in this repository rebuilds the graph from the upstream source. See the
README's Quick Start for the snapshot download and the from-source build.

## Known limitations

- Counts here are those stated by the repository README at the time this card was
  written; they are not re-measured by the card.
- Where a field above says *not recorded*, that is a gap in this repository rather
  than a property of the data.

## Links

| | |
|---|---|
| Samyama Graph | [github.com/samyama-ai/samyama-graph](https://github.com/samyama-ai/samyama-graph) |
| The Book | [samyama-ai.github.io/samyama-graph-book](https://samyama-ai.github.io/samyama-graph-book/) |
| IBM AssetOpsBench | [github.com/IBM/AssetOpsBench](https://github.com/IBM/AssetOpsBench) |
| Contact | [samyama.dev/contact](https://samyama.dev/contact) |
