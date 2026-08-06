# Getting Started — AssetOps Knowledge Graph

From `git clone` to your first answer. This KG is built from the **IBM AssetOpsBench** dataset, so the
working path today is **build-from-source** (see the note on the snapshot below).

---

## 1. Prerequisites

- **Python ≥ 3.10** (required by the `samyama` SDK; macOS ships 3.9 — use `python3.10`+).
- **git**
- **Docker** — to run the Samyama engine (HTTP `:8080`, RESP `:6379`).

## 2. Install

```bash
git clone https://github.com/samyama-ai/assetops-kg.git
cd assetops-kg
python3 -m venv .venv && source .venv/bin/activate     # Python >= 3.10
pip install -r requirements.txt                         # note: pulls torch (~large) for embeddings
```

## 3. Run the engine (Docker)

```bash
docker run --rm -p 8080:8080 -p 6379:6379 public.ecr.aws/f9f6l5u4/samyama-graph:1.1.0
```

## 4. Load the graph — into the `assetops` tenant

### Build from source (the working path)
The data comes from IBM's AssetOpsBench (139 scenarios, 9 asset classes) — clone it alongside this repo:

```bash
git clone https://github.com/IBM/AssetOpsBench.git ../AssetOpsBench

# create the tenant, then load (generates embeddings — needs torch)
curl -X POST http://localhost:8080/api/tenants -H 'Content-Type: application/json' \
  -d '{"id":"assetops","name":"AssetOps KG"}'
python -m etl.loader --data-dir ../AssetOpsBench --url http://localhost:8080 --graph assetops
```

### Snapshot (currently unavailable)
The README previously pointed at
`releases/download/kg-snapshots-v5/assetops.sgsnap`, but that asset **returns 404** — no snapshot is
published yet. Use the build-from-source path above until one is published (tracked upstream).

## 5. Ask your first question

Most critical equipment (by criticality score):

```bash
curl -s -X POST http://localhost:8080/api/query -H 'Content-Type: application/json' -d '{
  "graph": "assetops",
  "query": "MATCH (e:Equipment) RETURN e.name AS equipment, e.criticality_score AS crit ORDER BY crit DESC LIMIT 5"
}'
# → Chiller-1 (9), Chiller-2 (9), Chiller-3 (8), Chiller-4 (8), AHU-1 (7)
```

## 6. The ETL pipeline

- Data source: **IBM AssetOpsBench**.
- `etl/loader.py` — loads EAM-lite equipment, CouchDB sensor readings, FMSR failure modes and work
  orders, then generates embeddings. Run `python -m etl.loader --help`.
- Node labels: Equipment, Sensor, SensorReading, WorkOrder, FailureMode, SparePart.

## Next
- **[docs/QUERYING.md](docs/QUERYING.md)** — HTTP API and the Samyama CLI
