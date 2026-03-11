"""Asset query tools for the Industrial KG MCP Server."""


def register_asset_tools(mcp):
    @mcp.tool()
    def query_assets(asset_type: str, location: str | None = None) -> list[dict]:
        """Query industrial equipment by type and optional location.

        Returns a list of equipment nodes matching the given asset type (e.g.,
        'Compressor', 'Pump', 'Motor'). Optionally filter by location name to
        narrow results to a specific plant area or site.
        """
        from mcp_server.server import client, GRAPH

        if location:
            cypher = (
                "MATCH (e:Equipment)-[:LOCATED_IN]->(l:Location) "
                f"WHERE e.asset_type = '{asset_type}' AND l.name = '{location}' "
                "RETURN e.name, e.asset_type, e.status, e.criticality_score, "
                "e.mtbf_hours, l.name AS location"
            )
        else:
            cypher = (
                "MATCH (e:Equipment) "
                f"WHERE e.asset_type = '{asset_type}' "
                "RETURN e.name, e.asset_type, e.status, e.criticality_score, e.mtbf_hours"
            )

        result = client.query_readonly(cypher, GRAPH)
        assets = []
        for row in result.records:
            asset = {}
            for i, col in enumerate(result.columns):
                asset[col] = row[i]
            assets.append(asset)
        return assets

    @mcp.tool()
    def query_sensors(equipment_name: str) -> list[dict]:
        """Get all sensors attached to a piece of equipment.

        Returns sensor details including type, unit, and alarm thresholds
        (low_threshold, high_threshold) for the named equipment.
        """
        from mcp_server.server import client, GRAPH

        cypher = (
            "MATCH (s:Sensor)-[:MONITORS]->(e:Equipment) "
            f"WHERE e.name = '{equipment_name}' "
            "RETURN s.name, s.sensor_type, s.unit, "
            "s.low_threshold, s.high_threshold, s.status"
        )

        result = client.query_readonly(cypher, GRAPH)
        sensors = []
        for row in result.records:
            sensor = {}
            for i, col in enumerate(result.columns):
                sensor[col] = row[i]
            sensors.append(sensor)
        return sensors

    @mcp.tool()
    def query_sites() -> list[dict]:
        """Get the site hierarchy overview.

        Returns all sites with their locations and the count of equipment at
        each location. Useful for understanding the organizational structure
        of the industrial facility.
        """
        from mcp_server.server import client, GRAPH

        cypher = (
            "MATCH (s:Site)<-[:PART_OF]-(l:Location)<-[:LOCATED_IN]-(e:Equipment) "
            "RETURN s.name AS site, l.name AS location, count(e) AS equipment_count "
            "ORDER BY s.name, l.name"
        )

        result = client.query_readonly(cypher, GRAPH)
        sites = []
        for row in result.records:
            entry = {}
            for i, col in enumerate(result.columns):
                entry[col] = row[i]
            sites.append(entry)
        return sites
