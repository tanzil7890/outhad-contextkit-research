"""Causal chain query utilities for TCMGM."""
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def get_causal_chains_batch(
    graph: Any,
    event_ids: List[str],
    filters: dict,
    max_depth: int = 3,
    node_label: str = "",
) -> Dict[str, Dict[str, List[Dict]]]:
    r"""fetch forward + backward causal chains for many seeds in one round-trip.

    Replaces the N+1 pattern (``len(event_ids) * 2`` round-trips) with
    two ``UNWIND``-driven Cypher queries — one per direction — so a
    typical 3-seed exploration drops from 6 calls to 2.

    Args:
        graph: Neo4j graph instance (must expose ``.query(cypher, params)``).
        event_ids: Seed event names.
        filters: Same shape as :func:`get_causal_chain` (``user_id`` /
                 ``agent_id``).
        max_depth: Maximum chain depth.
        node_label: Optional node-label hint (e.g. ``":\`__Entity__\`"``).

    Returns:
        Mapping ``{event_id: {"forward": [...], "backward": [...]}}``.
        Missing seeds resolve to empty lists.
    """
    if not event_ids:
        return {}

    user_id = filters.get("user_id")
    agent_id = filters.get("agent_id")
    agent_clause = "AND start.agent_id = $agent_id" if agent_id else ""

    base_params: Dict[str, Any] = {
        "event_ids": list(event_ids),
        "user_id": user_id,
    }
    if agent_id:
        base_params["agent_id"] = agent_id

    # One round-trip per direction. Each query unwinds the seed list,
    # collects up to 10 chain rows per seed, and returns them grouped.
    forward_cypher = f"""
    UNWIND $event_ids AS seed_id
    MATCH (start {node_label} {{name: seed_id, user_id: $user_id}})
    WHERE 1=1 {agent_clause}
    OPTIONAL MATCH path = (start)-[:CAUSAL*1..{max_depth}]->(effect)
    WITH seed_id, path
    WHERE path IS NOT NULL
    RETURN seed_id,
           collect({{
               chain: nodes(path),
               links: relationships(path),
               depth: length(path)
           }})[0..10] AS chains
    """

    backward_cypher = f"""
    UNWIND $event_ids AS seed_id
    MATCH (start {node_label} {{name: seed_id, user_id: $user_id}})
    WHERE 1=1 {agent_clause}
    OPTIONAL MATCH path = (cause)-[:CAUSAL*1..{max_depth}]->(start)
    WITH seed_id, path
    WHERE path IS NOT NULL
    RETURN seed_id,
           collect({{
               chain: nodes(path),
               links: relationships(path),
               depth: length(path)
           }})[0..10] AS chains
    """

    result: Dict[str, Dict[str, List[Dict]]] = {
        eid: {"forward": [], "backward": []} for eid in event_ids
    }

    try:
        forward_rows = graph.query(forward_cypher, params=base_params) or []
        for row in forward_rows:
            seed_id = row.get("seed_id")
            chains = row.get("chains") or []
            if seed_id is not None:
                result.setdefault(
                    seed_id, {"forward": [], "backward": []}
                )["forward"] = [
                    {
                        "events": c.get("chain", []),
                        "causal_links": c.get("links", []),
                        "depth": c.get("depth", 0),
                    }
                    for c in chains
                ]
    except Exception as exc:
        logger.error("Batch forward causal-chain query failed: %s", exc, exc_info=True)

    try:
        backward_rows = graph.query(backward_cypher, params=base_params) or []
        for row in backward_rows:
            seed_id = row.get("seed_id")
            chains = row.get("chains") or []
            if seed_id is not None:
                result.setdefault(
                    seed_id, {"forward": [], "backward": []}
                )["backward"] = [
                    {
                        "events": c.get("chain", []),
                        "causal_links": c.get("links", []),
                        "depth": c.get("depth", 0),
                    }
                    for c in chains
                ]
    except Exception as exc:
        logger.error("Batch backward causal-chain query failed: %s", exc, exc_info=True)

    logger.info(
        "Batched causal-chain fetch: %d seeds → %d forward / %d backward chain rows",
        len(event_ids),
        sum(len(v["forward"]) for v in result.values()),
        sum(len(v["backward"]) for v in result.values()),
    )
    return result


def get_causal_chain(
    graph,
    start_event_id: str,
    filters: dict,
    max_depth: int = 5,
    direction: str = "forward",
    node_label: str = ""
) -> List[Dict]:
    """
    Get causal chain starting from an event.
    
    Args:
        graph: Neo4j graph instance
        start_event_id: Starting event ID (node name)
        filters: Query filters (user_id, agent_id, etc.)
        max_depth: Maximum chain depth (default: 5)
        direction: 'forward' (effects) or 'backward' (causes)
        node_label: Node label to use (e.g., ":`__Entity__`")
    
    Returns:
        List of dictionaries containing chain information
    """
    if direction not in ["forward", "backward"]:
        logger.error(f"Invalid direction '{direction}', must be 'forward' or 'backward'")
        return []
    
    # Build agent filter
    agent_filter = ""
    params = {
        "event_id": start_event_id,
        "user_id": filters.get("user_id"),
        "max_depth": max_depth
    }
    
    if filters.get("agent_id"):
        agent_filter = "AND start.agent_id = $agent_id"
        params["agent_id"] = filters.get("agent_id")
    
    if direction == "forward":
        # Get effects (what happened because of this)
        cypher = f"""
        MATCH (start {node_label} {{name: $event_id, user_id: $user_id}})
        WHERE 1=1 {agent_filter}
        MATCH path = (start)-[r:CAUSAL*1..{max_depth}]->(effect)
        RETURN nodes(path) as chain, relationships(path) as links, length(path) as depth
        ORDER BY depth
        LIMIT 10
        """
    else:
        # Get causes (what led to this)
        cypher = f"""
        MATCH (end {node_label} {{name: $event_id, user_id: $user_id}})
        WHERE 1=1 {agent_filter.replace('start', 'end')}
        MATCH path = (cause)-[r:CAUSAL*1..{max_depth}]->(end)
        RETURN nodes(path) as chain, relationships(path) as links, length(path) as depth
        ORDER BY depth
        LIMIT 10
        """
    
    try:
        results = graph.query(cypher, params=params)
        
        chains = []
        for result in results:
            chain = {
                "events": result.get("chain", []),
                "causal_links": result.get("links", []),
                "depth": result.get("depth", 0)
            }
            chains.append(chain)
        
        logger.info(f"Found {len(chains)} causal chains from '{start_event_id}' ({direction})")
        return chains
    
    except Exception as e:
        logger.error(f"Causal chain query failed: {e}", exc_info=True)
        return []


def find_root_causes(
    graph,
    effect_event_id: str,
    filters: dict,
    min_confidence: float = 0.7,
    node_label: str = ""
) -> List[Dict]:
    """
    Find root causes (events with no incoming causal links).
    
    Root causes are events that caused other events but were not themselves caused by anything.
    
    Args:
        graph: Neo4j graph instance
        effect_event_id: Effect event ID (node name)
        filters: Query filters (user_id, agent_id, etc.)
        min_confidence: Minimum confidence threshold for causal links (default: 0.7)
        node_label: Node label to use (e.g., ":`__Entity__`")
    
    Returns:
        List of root cause events
    """
    # Build agent filter
    agent_filter = ""
    params = {
        "event_id": effect_event_id,
        "user_id": filters.get("user_id"),
        "min_confidence": min_confidence
    }
    
    if filters.get("agent_id"):
        agent_filter = "AND effect.agent_id = $agent_id"
        params["agent_id"] = filters.get("agent_id")
    
    cypher = f"""
    MATCH (effect {node_label} {{name: $event_id, user_id: $user_id}})
    WHERE 1=1 {agent_filter}
    MATCH path = (root)-[r:CAUSAL*]->(effect)
    WHERE NOT ()-[:CAUSAL]->(root)
    AND all(rel in relationships(path) WHERE rel.confidence >= $min_confidence)
    RETURN DISTINCT root, length(path) as distance
    ORDER BY distance
    LIMIT 5
    """
    
    try:
        results = graph.query(cypher, params=params)
        
        root_causes = []
        for result in results:
            root_causes.append({
                "root": result.get("root", {}),
                "distance": result.get("distance", 0)
            })
        
        logger.info(f"Found {len(root_causes)} root causes for '{effect_event_id}'")
        return root_causes
    
    except Exception as e:
        logger.error(f"Root cause query failed: {e}", exc_info=True)
        return []


def get_causal_subgraph(
    graph,
    event_ids: List[str],
    filters: dict,
    max_depth: int = 3,
    node_label: str = ""
) -> Dict:
    """
    Get causal subgraph connecting multiple events.
    
    This finds all causal paths between the provided events.
    
    Args:
        graph: Neo4j graph instance
        event_ids: List of event IDs to connect
        filters: Query filters (user_id, agent_id, etc.)
        max_depth: Maximum path depth between events
        node_label: Node label to use
    
    Returns:
        Dictionary with nodes and relationships
    """
    if not event_ids or len(event_ids) < 2:
        logger.warning("Need at least 2 event IDs to build causal subgraph")
        return {"nodes": [], "relationships": []}
    
    # Build agent filter
    agent_filter = ""
    params = {
        "event_ids": event_ids,
        "user_id": filters.get("user_id"),
        "max_depth": max_depth
    }
    
    if filters.get("agent_id"):
        agent_filter = "AND source.agent_id = $agent_id AND target.agent_id = $agent_id"
        params["agent_id"] = filters.get("agent_id")
    
    cypher = f"""
    MATCH (source {node_label} {{user_id: $user_id}})
    WHERE source.name IN $event_ids {agent_filter.replace('target', 'source')}
    MATCH (target {node_label} {{user_id: $user_id}})
    WHERE target.name IN $event_ids {agent_filter}
    MATCH path = (source)-[r:CAUSAL*1..{max_depth}]-(target)
    WITH nodes(path) as path_nodes, relationships(path) as path_rels
    UNWIND path_nodes as node
    UNWIND path_rels as rel
    RETURN DISTINCT node, rel
    LIMIT 100
    """
    
    try:
        results = graph.query(cypher, params=params)
        
        nodes = []
        relationships = []
        
        for result in results:
            if result.get("node"):
                nodes.append(result["node"])
            if result.get("rel"):
                relationships.append(result["rel"])
        
        logger.info(f"Causal subgraph: {len(nodes)} nodes, {len(relationships)} relationships")
        return {
            "nodes": nodes,
            "relationships": relationships
        }
    
    except Exception as e:
        logger.error(f"Causal subgraph query failed: {e}", exc_info=True)
        return {"nodes": [], "relationships": []}

