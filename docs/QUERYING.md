# Querying the AssetOps KG

Ways to ask the graph questions, once it's loaded into the `assetops` tenant on a running engine
(see [GETTING_STARTED.md](../GETTING_STARTED.md)). The HTTP examples below were run live and return real
results.

---

## 1. HTTP API (`POST /api/query`) — recommended

Most critical equipment:

```bash
curl -s -X POST http://localhost:8080/api/query -H 'Content-Type: application/json' -d '{
  "graph": "assetops",
  "query": "MATCH (e:Equipment) RETURN e.name AS equipment, e.criticality_score AS crit ORDER BY crit DESC LIMIT 5"
}'
```
```json
{"columns":["equipment","crit"],
 "records":[["Chiller-1",9],["Chiller-2",9],["Chiller-3",8],["Chiller-4",8],["AHU-1",7]]}
```

Highest-severity failure modes:

```bash
curl -s -X POST http://localhost:8080/api/query -H 'Content-Type: application/json' -d '{
  "graph": "assetops",
  "query": "MATCH (fm:FailureMode) RETURN fm.name AS failure_mode, fm.severity AS severity ORDER BY fm.severity DESC LIMIT 5"
}'
# → Compressor-Overheating-Failed (high), Heat-Exchangers-Fans (medium), Evaporator-Water-side (medium), ...
```

## 2. Samyama CLI (Redis wire protocol, `:6379`)

```bash
redis-cli -p 6379 GRAPH.QUERY assetops \
  "MATCH (e:Equipment) RETURN e.name, e.criticality_score AS crit ORDER BY crit DESC LIMIT 5"
```

> **Note:** on the current engine build the RESP/`GRAPH.QUERY` path returns empty results for some
> queries on this tenant that the HTTP API answers correctly (e.g. `count(Equipment)` → HTTP 20, RESP 0) —
> same class of issue as [samyama-graph#334](https://github.com/samyama-ai/samyama-graph/issues/334).
> Until that's fixed, prefer the **HTTP API**.

## 3. MCP

`mcp_server/` ships tool definitions, but the current `server.py` starts an **empty embedded graph** and
has no `--url` option, so its tools return nothing against a loaded engine. Tracked in this repo's issues;
use the HTTP API meanwhile.

---

## More queries
See the `README` "Example Queries" section (dependency chains, sensor coverage) and `mcp_server/tools/`
for the graph-powered analytics the tools are built around.
