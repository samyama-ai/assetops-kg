"""Analytics tools for the Industrial KG MCP Server."""


def register_analytics_tools(mcp):
    @mcp.tool()
    def criticality_ranking(top_n: int = 10) -> list[dict]:
        """Rank equipment by structural criticality using PageRank.

        Runs the PageRank algorithm on the equipment dependency graph to
        identify the most structurally critical assets. Equipment with high
        PageRank scores are central to many dependency chains and represent
        high-impact failure points.

        Args:
            top_n: Number of top-ranked equipment to return (default 10).
        """
        from mcp_server.server import client, GRAPH

        # Run PageRank on Equipment nodes connected by DEPENDS_ON edges
        scores = client.page_rank(
            label="Equipment",
            edge_type="DEPENDS_ON",
            damping=0.85,
            iterations=20,
            tolerance=1e-6,
        )

        # Sort by score descending and take top N
        ranked_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

        # Enrich with equipment details
        ranking = []
        for rank, (node_id, score) in enumerate(ranked_ids, start=1):
            cypher = (
                f"MATCH (e:Equipment) WHERE id(e) = {node_id} "
                "RETURN e.name, e.asset_type, e.status, e.criticality_score, e.mtbf_hours"
            )
            detail = client.query_readonly(cypher, GRAPH)
            if detail.records:
                row = detail.records[0]
                entry = {
                    "rank": rank,
                    "node_id": node_id,
                    "pagerank_score": round(score, 6),
                }
                for i, col in enumerate(detail.columns):
                    entry[col] = row[i]
                ranking.append(entry)

        return ranking

    @mcp.tool()
    def maintenance_clusters(location: str | None = None) -> list[dict]:
        """Group equipment into maintenance urgency clusters.

        Classifies equipment into urgency tiers based on criticality_score
        and mtbf_hours (mean time between failures). Optionally filter by
        location. Returns clusters sorted by urgency (critical first).

        Urgency tiers:
        - critical: criticality_score >= 0.8 and mtbf_hours < 2000
        - high: criticality_score >= 0.6 or mtbf_hours < 4000
        - medium: criticality_score >= 0.3
        - low: everything else
        """
        from mcp_server.server import client, GRAPH

        if location:
            cypher = (
                "MATCH (e:Equipment)-[:LOCATED_IN]->(l:Location) "
                f"WHERE l.name = '{location}' "
                "RETURN e.name, e.asset_type, e.criticality_score, "
                "e.mtbf_hours, e.status, l.name AS location"
            )
        else:
            cypher = (
                "MATCH (e:Equipment) "
                "RETURN e.name, e.asset_type, e.criticality_score, "
                "e.mtbf_hours, e.status"
            )

        result = client.query_readonly(cypher, GRAPH)

        clusters = {
            "critical": [],
            "high": [],
            "medium": [],
            "low": [],
        }

        for row in result.records:
            equipment = {}
            for i, col in enumerate(result.columns):
                equipment[col] = row[i]

            criticality = equipment.get("e.criticality_score") or 0
            mtbf = equipment.get("e.mtbf_hours") or float("inf")

            if criticality >= 0.8 and mtbf < 2000:
                tier = "critical"
            elif criticality >= 0.6 or mtbf < 4000:
                tier = "high"
            elif criticality >= 0.3:
                tier = "medium"
            else:
                tier = "low"

            equipment["urgency_tier"] = tier
            clusters[tier].append(equipment)

        # Return as a list of cluster dicts, sorted by urgency
        return [
            {
                "tier": tier,
                "count": len(items),
                "equipment": items,
            }
            for tier, items in clusters.items()
            if items  # omit empty tiers
        ]
