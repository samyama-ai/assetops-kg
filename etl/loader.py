"""Main ETL orchestrator for loading AssetOpsBench data into Samyama KG."""

import click
from rich.console import Console
from samyama import SamyamaClient

from .eamlite_loader import load_eamlite
from .couchdb_loader import load_couchdb
from .fmsr_loader import load_fmsr
from .embedding_gen import generate_embeddings

console = Console()


@click.command()
@click.option("--data-dir", required=True, help="Path to AssetOpsBench data directory")
@click.option("--graph", default="industrial", help="Graph name")
@click.option("--embed-model", default="all-MiniLM-L6-v2", help="Sentence-transformer model")
def main(data_dir: str, graph: str, embed_model: str):
    """Load AssetOpsBench data into Samyama Industrial Knowledge Graph."""
    console.print("[bold green]Industrial KG ETL Pipeline[/bold green]")

    client = SamyamaClient.embedded()

    # Step 1: Load site/location/equipment hierarchy from EAMLite
    console.print("\n[bold]Step 1:[/bold] Loading asset hierarchy (EAMLite)...")
    eam_stats = load_eamlite(client, data_dir, graph)
    console.print(
        f"  Sites: {eam_stats['sites']}, "
        f"Locations: {eam_stats['locations']}, "
        f"Equipment: {eam_stats['equipment']}"
    )

    # Step 2: Load sensor data from CouchDB JSON
    console.print("\n[bold]Step 2:[/bold] Loading sensor data (CouchDB)...")
    sensor_stats = load_couchdb(client, data_dir, graph)
    console.print(
        f"  Sensors: {sensor_stats['sensors']}, "
        f"Readings: {sensor_stats['readings']}"
    )

    # Step 3: Load failure modes from FMSR YAML
    console.print("\n[bold]Step 3:[/bold] Loading failure modes (FMSR)...")
    fmsr_stats = load_fmsr(client, data_dir, graph)
    console.print(
        f"  Failure modes: {fmsr_stats['failure_modes']}, "
        f"MONITORS edges: {fmsr_stats['monitors_edges']}"
    )

    # Step 4: Generate embeddings for failure modes and anomalies
    console.print("\n[bold]Step 4:[/bold] Generating embeddings...")
    embed_stats = generate_embeddings(client, graph, embed_model)
    console.print(
        f"  Embedded: {embed_stats['embedded_nodes']} nodes "
        f"({embed_stats['dimensions']}-dim)"
    )

    # Summary
    status = client.status()
    console.print(
        f"\n[bold green]ETL Complete:[/bold green] "
        f"{status.nodes} nodes, {status.edges} edges"
    )


if __name__ == "__main__":
    main()
