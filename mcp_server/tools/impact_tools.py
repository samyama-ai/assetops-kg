"""Impact and dependency analysis tools for the Industrial KG MCP Server."""


def register_impact_tools(mcp):
    @mcp.tool()
    def impact_analysis(equipment_name: str) -> dict:
        """Analyze the cascade impact if a piece of equipment fails.

        Performs a BFS traversal over reversed DEPENDS_ON edges to discover
        all downstream equipment that would be affected by a failure of the
        named equipment. Returns the cascade tree with depth levels indicating
        how many hops away each affected asset is.
        """
        from mcp_server.server import client, GRAPH

        # Find the source equipment node ID
        cypher = (
            "MATCH (e:Equipment) "
            f"WHERE e.name = '{equipment_name}' "
            "RETURN id(e), e.name, e.asset_type"
        )
        result = client.query_readonly(cypher, GRAPH)
        if not result.records:
            return {"error": f"Equipment '{equipment_name}' not found", "affected": []}

        source_id = result.records[0][0]

        # BFS traversal: find all equipment that depends ON this one
        # (reversed DEPENDS_ON means: who depends on me?)
        affected = []
        visited = {source_id}
        frontier = [source_id]
        depth = 0

        while frontier:
            depth += 1
            next_frontier = []
            for node_id in frontier:
                # Find nodes that depend on current node (reverse direction)
                dep_cypher = (
                    f"MATCH (dep:Equipment)-[:DEPENDS_ON]->(e:Equipment) "
                    f"WHERE id(e) = {node_id} "
                    "RETURN id(dep), dep.name, dep.asset_type, dep.criticality_score"
                )
                dep_result = client.query_readonly(dep_cypher, GRAPH)
                for row in dep_result.records:
                    dep_id = row[0]
                    if dep_id not in visited:
                        visited.add(dep_id)
                        next_frontier.append(dep_id)
                        affected.append({
                            "node_id": dep_id,
                            "name": row[1],
                            "asset_type": row[2],
                            "criticality_score": row[3],
                            "cascade_depth": depth,
                        })
            frontier = next_frontier

        return {
            "source": equipment_name,
            "source_id": source_id,
            "total_affected": len(affected),
            "max_cascade_depth": depth - 1 if depth > 1 else 0,
            "affected": affected,
        }

    @mcp.tool()
    def dependency_chain(equipment_name: str) -> dict:
        """Find what a piece of equipment depends on.

        Traverses DEPENDS_ON edges forward from the named equipment to
        discover its upstream dependencies. Returns the full dependency
        list with depth levels.
        """
        from mcp_server.server import client, GRAPH

        # Find the source equipment node ID
        cypher = (
            "MATCH (e:Equipment) "
            f"WHERE e.name = '{equipment_name}' "
            "RETURN id(e), e.name, e.asset_type"
        )
        result = client.query_readonly(cypher, GRAPH)
        if not result.records:
            return {"error": f"Equipment '{equipment_name}' not found", "dependencies": []}

        source_id = result.records[0][0]

        # BFS traversal: follow DEPENDS_ON edges forward
        dependencies = []
        visited = {source_id}
        frontier = [source_id]
        depth = 0

        while frontier:
            depth += 1
            next_frontier = []
            for node_id in frontier:
                dep_cypher = (
                    f"MATCH (e:Equipment)-[:DEPENDS_ON]->(dep:Equipment) "
                    f"WHERE id(e) = {node_id} "
                    "RETURN id(dep), dep.name, dep.asset_type, dep.criticality_score"
                )
                dep_result = client.query_readonly(dep_cypher, GRAPH)
                for row in dep_result.records:
                    dep_id = row[0]
                    if dep_id not in visited:
                        visited.add(dep_id)
                        next_frontier.append(dep_id)
                        dependencies.append({
                            "node_id": dep_id,
                            "name": row[1],
                            "asset_type": row[2],
                            "criticality_score": row[3],
                            "dependency_depth": depth,
                        })
            frontier = next_frontier

        return {
            "source": equipment_name,
            "source_id": source_id,
            "total_dependencies": len(dependencies),
            "max_depth": depth - 1 if depth > 1 else 0,
            "dependencies": dependencies,
        }
