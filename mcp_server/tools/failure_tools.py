"""Failure mode tools for the Industrial KG MCP Server."""


def register_failure_tools(mcp):
    @mcp.tool()
    def find_similar_failures(description: str, k: int = 5) -> list[dict]:
        """Find failure modes semantically similar to a description.

        Uses vector similarity search on FailureMode embeddings to find the
        top-k most relevant historical failure modes. Useful for diagnosing
        new issues by finding past failures with similar symptoms or root
        causes.

        Args:
            description: Natural language description of the failure or symptom.
            k: Number of similar failure modes to return (default 5).
        """
        from mcp_server.server import client, GRAPH
        from sentence_transformers import SentenceTransformer

        # Encode the description into a vector
        model = SentenceTransformer("all-MiniLM-L6-v2")
        query_vector = model.encode(description).tolist()

        # Search the FailureMode vector index
        results = client.vector_search("FailureMode", "embedding", query_vector, k)

        # Enrich results with failure mode details
        failures = []
        for node_id, distance in results:
            cypher = (
                f"MATCH (f:FailureMode) WHERE id(f) = {node_id} "
                "RETURN f.name, f.description, f.severity, f.category"
            )
            detail = client.query_readonly(cypher, GRAPH)
            if detail.records:
                row = detail.records[0]
                failure = {
                    "node_id": node_id,
                    "distance": float(distance),
                }
                for i, col in enumerate(detail.columns):
                    failure[col] = row[i]
                failures.append(failure)

        return failures

    @mcp.tool()
    def query_failure_modes(
        equipment_name: str | None = None, severity: str | None = None
    ) -> list[dict]:
        """Query failure modes, optionally filtering by equipment or severity.

        Returns failure modes from the knowledge graph. Can filter by the name
        of the equipment they affect, by severity level (e.g., 'Critical',
        'High', 'Medium', 'Low'), or both.
        """
        from mcp_server.server import client, GRAPH

        if equipment_name and severity:
            cypher = (
                "MATCH (f:FailureMode)-[:AFFECTS]->(e:Equipment) "
                f"WHERE e.name = '{equipment_name}' AND f.severity = '{severity}' "
                "RETURN f.name, f.description, f.severity, f.category, e.name AS equipment"
            )
        elif equipment_name:
            cypher = (
                "MATCH (f:FailureMode)-[:AFFECTS]->(e:Equipment) "
                f"WHERE e.name = '{equipment_name}' "
                "RETURN f.name, f.description, f.severity, f.category, e.name AS equipment"
            )
        elif severity:
            cypher = (
                "MATCH (f:FailureMode) "
                f"WHERE f.severity = '{severity}' "
                "RETURN f.name, f.description, f.severity, f.category"
            )
        else:
            cypher = (
                "MATCH (f:FailureMode) "
                "RETURN f.name, f.description, f.severity, f.category"
            )

        result = client.query_readonly(cypher, GRAPH)
        failures = []
        for row in result.records:
            failure = {}
            for i, col in enumerate(result.columns):
                failure[col] = row[i]
            failures.append(failure)
        return failures
