"""Industrial Knowledge Graph MCP Server.

Exposes graph-powered tools for industrial asset operations via MCP protocol.
"""
from fastmcp import FastMCP
from samyama import SamyamaClient

from .tools.asset_tools import register_asset_tools
from .tools.failure_tools import register_failure_tools
from .tools.impact_tools import register_impact_tools
from .tools.analytics_tools import register_analytics_tools

mcp = FastMCP("Industrial KG")

# Global client — initialized on startup
client: SamyamaClient | None = None
GRAPH = "industrial"


@mcp.on_startup
async def startup():
    global client
    client = SamyamaClient.embedded()


# Register all tool groups
register_asset_tools(mcp)
register_failure_tools(mcp)
register_impact_tools(mcp)
register_analytics_tools(mcp)

if __name__ == "__main__":
    mcp.run()
