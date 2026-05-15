import logging

from outhad_contextkit.memory.utils import format_entities

try:
    from langchain_neo4j import Neo4jGraph
except ImportError:
    raise ImportError("langchain_neo4j is not installed. Please install it using pip install langchain-neo4j")

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("rank_bm25 is not installed. Please install it using pip install rank-bm25")

from outhad_contextkit.graphs.tools import (
    DELETE_MEMORY_STRUCT_TOOL_GRAPH,
    DELETE_MEMORY_TOOL_GRAPH,
    EXTRACT_ENTITIES_STRUCT_TOOL,
    EXTRACT_ENTITIES_TOOL,
    RELATIONS_STRUCT_TOOL,
    RELATIONS_TOOL,
)
from outhad_contextkit.graphs.utils import EXTRACT_RELATIONS_PROMPT, get_delete_messages
from outhad_contextkit.utils.factory import EmbedderFactory, LlmFactory

logger = logging.getLogger(__name__)


class MemoryGraph:
    def __init__(self, config):
        self.config = config
        # Phase PT1 — pass through pool tuning when callers set it on
        # Neo4jConfig. ``driver_config`` is forwarded verbatim to the
        # underlying ``neo4j.GraphDatabase.driver(...)`` so each kwarg
        # below is the official driver kwarg name.
        driver_config = {"notifications_min_severity": "OFF"}
        cfg = self.config.graph_store.config
        for kwarg in (
            "max_connection_pool_size",
            "connection_acquisition_timeout",
            "max_connection_lifetime",
        ):
            value = getattr(cfg, kwarg, None)
            if value is not None:
                driver_config[kwarg] = value

        self.graph = Neo4jGraph(
            self.config.graph_store.config.url,
            self.config.graph_store.config.username,
            self.config.graph_store.config.password,
            self.config.graph_store.config.database,
            refresh_schema=False,
            driver_config=driver_config,
        )
        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider, self.config.embedder.config, self.config.vector_store.config
        )
        self.node_label = ":`__Entity__`" if self.config.graph_store.config.base_label else ""

        if self.config.graph_store.config.base_label:
            # Safely add user_id index
            try:
                self.graph.query(f"CREATE INDEX entity_single IF NOT EXISTS FOR (n {self.node_label}) ON (n.user_id)")
            except Exception:
                pass
            try:  # Safely try to add composite index (Enterprise only)
                self.graph.query(
                    f"CREATE INDEX entity_composite IF NOT EXISTS FOR (n {self.node_label}) ON (n.name, n.user_id)"
                )
            except Exception:
                pass

        self.llm_provider = "openai_structured"
        if self.config.llm.provider:
            self.llm_provider = self.config.llm.provider
        if self.config.graph_store.llm:
            self.llm_provider = self.config.graph_store.llm.provider

        self.llm = LlmFactory.create(self.llm_provider, self.config.llm.config)
        self.user_id = None
        self.threshold = 0.7

        # Initialize TCMGM temporal schema
        self._create_temporal_schema()
    
    def _create_temporal_schema(self):
        """Create temporal indexes and constraints for TCMGM support.

        Phase PT5 — every TCMGM query path that filters by timestamp,
        causal_type, or (user_id, timestamp) is index-backed. Statements
        are ``CREATE INDEX ... IF NOT EXISTS`` so repeat boots are no-ops.
        Each is wrapped in its own try/except — a backend that does not
        support a given index syntax (older Neo4j, Memgraph variants)
        keeps booting instead of crashing.
        """
        index_specs = (
            # name,                    columns
            ("event_timestamp",        "(n.timestamp)"),
            ("event_modality",         "(n.modality)"),
            ("event_confidence",       "(n.confidence)"),
            # PT5 — causal-query hot paths.
            ("event_causal_type",      "(n.causal_type)"),
            ("event_causal_strength",  "(n.causal_strength)"),
            # PT5 — composite for windowed user-scoped queries: WHERE
            # user_id = $u AND timestamp >= $t1 AND timestamp <= $t2.
            ("event_user_timestamp",   "(n.user_id, n.timestamp)"),
            # PT5 — direct event_id lookup (used by causal_queries
            # batch UNWIND in P4 and reference tracking in D6).
            ("event_id_lookup",        "(n.event_id)"),
        )
        created = 0
        for name, cols in index_specs:
            try:
                self.graph.query(
                    f"CREATE INDEX {name} IF NOT EXISTS FOR (n {self.node_label}) ON {cols}"
                )
                created += 1
            except Exception:
                # Backend may not support this index shape (e.g.
                # composite indexes pre-4.x, or non-Neo4j drivers).
                pass

        logger.info("TCMGM temporal schema initialized (%d/%d indexes)", created, len(index_specs))

    def add(self, data, filters, timestamp=None, confidence=1.0, modality="text", 
            image_hash=None, audio_hash=None, metadata=None):
        """
        Adds data to the graph with temporal and multimodal attributes (TCMGM support).

        Args:
            data (str): The data to add to the graph.
            filters (dict): A dictionary containing filters to be applied during the addition.
            timestamp (datetime, optional): Event timestamp. Defaults to now.
            confidence (float, optional): Confidence score (0-1). Defaults to 1.0.
            modality (str, optional): Modality type (text, image, audio, video). Defaults to "text".
            image_hash (str, optional): Hash for image modality.
            audio_hash (str, optional): Hash for audio modality.
            metadata (dict, optional): Additional metadata for temporal events.
        """
        # Add temporal attributes to filters for TCMGM support
        if timestamp is None:
            from datetime import datetime
            timestamp = datetime.utcnow()
        
        if metadata is None:
            metadata = {}
        
        # Store temporal attributes in filters for downstream processing
        filters['timestamp'] = timestamp.isoformat()
        filters['confidence'] = confidence
        filters['modality'] = modality
        if image_hash:
            filters['image_hash'] = image_hash
        if audio_hash:
            filters['audio_hash'] = audio_hash
        # Merge metadata
        if 'metadata' not in filters:
            filters['metadata'] = {}
        filters['metadata'].update(metadata)
        
        entity_type_map = self._retrieve_nodes_from_data(data, filters)
        to_be_added = self._establish_nodes_relations_from_data(data, filters, entity_type_map)
        search_output = self._search_graph_db(node_list=list(entity_type_map.keys()), filters=filters)
        to_be_deleted = self._get_delete_entities_from_search_output(search_output, data, filters)

        # TODO: Batch queries with APOC plugin
        # TODO: Add more filter support
        deleted_entities = self._delete_entities(to_be_deleted, filters)
        added_entities = self._add_entities(to_be_added, filters, entity_type_map)

        return {"deleted_entities": deleted_entities, "added_entities": added_entities}

    def search(self, query, filters, limit=100):
        """
        Search for memories and related graph data.

        Args:
            query (str): Query to search for.
            filters (dict): A dictionary containing filters to be applied during the search.
            limit (int): The maximum number of nodes and relationships to retrieve. Defaults to 100.

        Returns:
            dict: A dictionary containing:
                - "contexts": List of search results from the base data store.
                - "entities": List of related graph data based on the query.
        """
        entity_type_map = self._retrieve_nodes_from_data(query, filters)
        search_output = self._search_graph_db(node_list=list(entity_type_map.keys()), filters=filters)

        if not search_output:
            return []

        search_outputs_sequence = [
            [item["source"], item["relationship"], item["destination"]] for item in search_output
        ]
        bm25 = BM25Okapi(search_outputs_sequence)

        tokenized_query = query.split(" ")
        reranked_results = bm25.get_top_n(tokenized_query, search_outputs_sequence, n=5)

        search_results = []
        for item in reranked_results:
            search_results.append({"source": item[0], "relationship": item[1], "destination": item[2]})

        logger.info(f"Returned {len(search_results)} search results")

        return search_results

    def delete_all(self, filters):
        if filters.get("agent_id"):
            cypher = f"""
            MATCH (n {self.node_label} {{user_id: $user_id, agent_id: $agent_id}})
            DETACH DELETE n
            """
            params = {"user_id": filters["user_id"], "agent_id": filters["agent_id"]}
        else:
            cypher = f"""
            MATCH (n {self.node_label} {{user_id: $user_id}})
            DETACH DELETE n
            """
            params = {"user_id": filters["user_id"]}
        self.graph.query(cypher, params=params)

    def get_all(self, filters, limit=100):
        """
        Retrieves all nodes and relationships from the graph database based on optional filtering criteria.
         Args:
            filters (dict): A dictionary containing filters to be applied during the retrieval.
            limit (int): The maximum number of nodes and relationships to retrieve. Defaults to 100.
        Returns:
            list: A list of dictionaries, each containing:
                - 'contexts': The base data store response for each memory.
                - 'entities': A list of strings representing the nodes and relationships
        """
        agent_filter = ""
        params = {"user_id": filters["user_id"], "limit": limit}
        if filters.get("agent_id"):
            agent_filter = "AND n.agent_id = $agent_id AND m.agent_id = $agent_id"
            params["agent_id"] = filters["agent_id"]

        query = f"""
        MATCH (n {self.node_label} {{user_id: $user_id}})-[r]->(m {self.node_label} {{user_id: $user_id}})
        WHERE 1=1 {agent_filter}
        RETURN n.name AS source, type(r) AS relationship, m.name AS target
        LIMIT $limit
        """
        results = self.graph.query(query, params=params)

        final_results = []
        for result in results:
            final_results.append(
                {
                    "source": result["source"],
                    "relationship": result["relationship"],
                    "target": result["target"],
                }
            )

        logger.info(f"Retrieved {len(final_results)} relationships")

        return final_results

    def _retrieve_nodes_from_data(self, data, filters):
        """Extracts all the entities mentioned in the query."""
        _tools = [EXTRACT_ENTITIES_TOOL]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [EXTRACT_ENTITIES_STRUCT_TOOL]
        search_results = self.llm.generate_response(
            messages=[
                {
                    "role": "system",
                    "content": f"You are a smart assistant who understands entities and their types in a given text. If user message contains self reference such as 'I', 'me', 'my' etc. then use {filters['user_id']} as the source entity. Extract all the entities from the text. ***DO NOT*** answer the question itself if the given text is a question.",
                },
                {"role": "user", "content": data},
            ],
            tools=_tools,
        )

        entity_type_map = {}

        try:
            for tool_call in search_results["tool_calls"]:
                if tool_call["name"] != "extract_entities":
                    continue
                for item in tool_call["arguments"]["entities"]:
                    entity_type_map[item["entity"]] = item["entity_type"]
        except Exception as e:
            logger.exception(
                f"Error in search tool: {e}, llm_provider={self.llm_provider}, search_results={search_results}"
            )

        entity_type_map = {k.lower().replace(" ", "_"): v.lower().replace(" ", "_") for k, v in entity_type_map.items()}
        logger.debug(f"Entity type map: {entity_type_map}\n search_results={search_results}")
        return entity_type_map

    def _establish_nodes_relations_from_data(self, data, filters, entity_type_map):
        """Establish relations among the extracted nodes."""

        # Compose user identification string for prompt
        user_identity = f"user_id: {filters['user_id']}"
        if filters.get("agent_id"):
            user_identity += f", agent_id: {filters['agent_id']}"

        if self.config.graph_store.custom_prompt:
            system_content = EXTRACT_RELATIONS_PROMPT.replace("USER_ID", user_identity)
            # Add the custom prompt line if configured
            system_content = system_content.replace("CUSTOM_PROMPT", f"4. {self.config.graph_store.custom_prompt}")
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": data},
            ]
        else:
            system_content = EXTRACT_RELATIONS_PROMPT.replace("USER_ID", user_identity)
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"List of entities: {list(entity_type_map.keys())}. \n\nText: {data}"},
            ]

        _tools = [RELATIONS_TOOL]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [RELATIONS_STRUCT_TOOL]

        extracted_entities = self.llm.generate_response(
            messages=messages,
            tools=_tools,
        )

        entities = []
        if extracted_entities.get("tool_calls"):
            entities = extracted_entities["tool_calls"][0].get("arguments", {}).get("entities", [])

        entities = self._remove_spaces_from_entities(entities)
        logger.debug(f"Extracted entities: {entities}")
        return entities

    def _search_graph_db(self, node_list, filters, limit=100):
        """Search similar nodes among and their respective incoming and outgoing relations."""
        result_relations = []
        agent_filter = ""
        if filters.get("agent_id"):
            agent_filter = "AND n.agent_id = $agent_id AND m.agent_id = $agent_id"

        for node in node_list:
            n_embedding = self.embedding_model.embed(node)

            cypher_query = f"""
            MATCH (n {self.node_label})
            WHERE n.embedding IS NOT NULL AND n.user_id = $user_id
            {agent_filter}
            WITH n, round(2 * vector.similarity.cosine(n.embedding, $n_embedding) - 1, 4) AS similarity // denormalize for backward compatibility
            WHERE similarity >= $threshold
            CALL {{
                MATCH (n)-[r]->(m)
                WHERE m.user_id = $user_id {agent_filter.replace("n.", "m.")} 
                RETURN n.name AS source, elementId(n) AS source_id, type(r) AS relationship, elementId(r) AS relation_id, m.name AS destination, elementId(m) AS destination_id
                UNION
                MATCH (m)-[r]->(n)
                WHERE m.user_id = $user_id {agent_filter.replace("n.", "m.")}
                RETURN m.name AS source, elementId(m) AS source_id, type(r) AS relationship, elementId(r) AS relation_id, n.name AS destination, elementId(n) AS destination_id
            }}
            WITH distinct source, source_id, relationship, relation_id, destination, destination_id, similarity
            RETURN source, source_id, relationship, relation_id, destination, destination_id, similarity
            ORDER BY similarity DESC
            LIMIT $limit
            """

            params = {
                "n_embedding": n_embedding,
                "threshold": self.threshold,
                "user_id": filters["user_id"],
                "limit": limit,
            }
            if filters.get("agent_id"):
                params["agent_id"] = filters["agent_id"]

            ans = self.graph.query(cypher_query, params=params)
            result_relations.extend(ans)

        return result_relations

    def _get_delete_entities_from_search_output(self, search_output, data, filters):
        """Get the entities to be deleted from the search output."""
        search_output_string = format_entities(search_output)

        # Compose user identification string for prompt
        user_identity = f"user_id: {filters['user_id']}"
        if filters.get("agent_id"):
            user_identity += f", agent_id: {filters['agent_id']}"

        system_prompt, user_prompt = get_delete_messages(search_output_string, data, user_identity)

        _tools = [DELETE_MEMORY_TOOL_GRAPH]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [
                DELETE_MEMORY_STRUCT_TOOL_GRAPH,
            ]

        memory_updates = self.llm.generate_response(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=_tools,
        )

        to_be_deleted = []
        for item in memory_updates.get("tool_calls", []):
            if item.get("name") == "delete_graph_memory":
                to_be_deleted.append(item.get("arguments"))
        # Clean entities formatting
        to_be_deleted = self._remove_spaces_from_entities(to_be_deleted)
        logger.debug(f"Deleted relationships: {to_be_deleted}")
        return to_be_deleted

    def _delete_entities(self, to_be_deleted, filters):
        """Delete the entities from the graph."""
        user_id = filters["user_id"]
        agent_id = filters.get("agent_id", None)
        results = []

        for item in to_be_deleted:
            source = item["source"]
            destination = item["destination"]
            relationship = item["relationship"]

            # Build the agent filter for the query
            agent_filter = ""
            params = {
                "source_name": source,
                "dest_name": destination,
                "user_id": user_id,
            }

            if agent_id:
                agent_filter = "AND n.agent_id = $agent_id AND m.agent_id = $agent_id"
                params["agent_id"] = agent_id

            # Delete the specific relationship between nodes
            cypher = f"""
            MATCH (n {self.node_label} {{name: $source_name, user_id: $user_id}})
            -[r:{relationship}]->
            (m {self.node_label} {{name: $dest_name, user_id: $user_id}})
            WHERE 1=1 {agent_filter}
            DELETE r
            RETURN 
                n.name AS source,
                m.name AS target,
                type(r) AS relationship
            """

            result = self.graph.query(cypher, params=params)
            results.append(result)

        return results

    def _add_entities(self, to_be_added, filters, entity_type_map):
        """Add the new entities to the graph. Merge the nodes if they already exist."""
        user_id = filters["user_id"]
        agent_id = filters.get("agent_id", None)
        
        # Extract temporal attributes from filters for TCMGM support
        timestamp = filters.get("timestamp")
        confidence = filters.get("confidence", 1.0)
        modality = filters.get("modality", "text")
        image_hash = filters.get("image_hash")
        audio_hash = filters.get("audio_hash")
        
        results = []
        for item in to_be_added:
            # entities
            source = item["source"]
            destination = item["destination"]
            relationship = item["relationship"]

            # types
            source_type = entity_type_map.get(source, "__User__")
            source_label = self.node_label if self.node_label else f":`{source_type}`"
            source_extra_set = f", source:`{source_type}`" if self.node_label else ""
            destination_type = entity_type_map.get(destination, "__User__")
            destination_label = self.node_label if self.node_label else f":`{destination_type}`"
            destination_extra_set = f", destination:`{destination_type}`" if self.node_label else ""

            # embeddings
            source_embedding = self.embedding_model.embed(source)
            dest_embedding = self.embedding_model.embed(destination)

            # search for the nodes with the closest embeddings
            source_node_search_result = self._search_source_node(source_embedding, filters, threshold=0.9)
            destination_node_search_result = self._search_destination_node(dest_embedding, filters, threshold=0.9)

            # TODO: Create a cypher query and common params for all the cases
            if not destination_node_search_result and source_node_search_result:
                # Build destination MERGE properties
                merge_props = ["name: $destination_name", "user_id: $user_id"]
                if agent_id:
                    merge_props.append("agent_id: $agent_id")
                merge_props_str = ", ".join(merge_props)

                cypher = f"""
                MATCH (source)
                WHERE elementId(source) = $source_id
                SET source.mentions = coalesce(source.mentions, 0) + 1
                WITH source
                MERGE (destination {destination_label} {{{merge_props_str}}})
                ON CREATE SET
                    destination.created = timestamp(),
                    destination.mentions = 1
                    {destination_extra_set}
                ON MATCH SET
                    destination.mentions = coalesce(destination.mentions, 0) + 1
                WITH source, destination
                CALL db.create.setNodeVectorProperty(destination, 'embedding', $destination_embedding)
                WITH source, destination
                MERGE (source)-[r:{relationship}]->(destination)
                ON CREATE SET 
                    r.created = timestamp(),
                    r.mentions = 1
                ON MATCH SET
                    r.mentions = coalesce(r.mentions, 0) + 1
                RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                """

                params = {
                    "source_id": source_node_search_result[0]["elementId(source_candidate)"],
                    "destination_name": destination,
                    "destination_embedding": dest_embedding,
                    "user_id": user_id,
                }
                if agent_id:
                    params["agent_id"] = agent_id

            elif destination_node_search_result and not source_node_search_result:
                # Build source MERGE properties
                merge_props = ["name: $source_name", "user_id: $user_id"]
                if agent_id:
                    merge_props.append("agent_id: $agent_id")
                merge_props_str = ", ".join(merge_props)

                cypher = f"""
                MATCH (destination)
                WHERE elementId(destination) = $destination_id
                SET destination.mentions = coalesce(destination.mentions, 0) + 1
                WITH destination
                MERGE (source {source_label} {{{merge_props_str}}})
                ON CREATE SET
                    source.created = timestamp(),
                    source.mentions = 1
                    {source_extra_set}
                ON MATCH SET
                    source.mentions = coalesce(source.mentions, 0) + 1
                WITH source, destination
                CALL db.create.setNodeVectorProperty(source, 'embedding', $source_embedding)
                WITH source, destination
                MERGE (source)-[r:{relationship}]->(destination)
                ON CREATE SET 
                    r.created = timestamp(),
                    r.mentions = 1
                ON MATCH SET
                    r.mentions = coalesce(r.mentions, 0) + 1
                RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                """

                params = {
                    "destination_id": destination_node_search_result[0]["elementId(destination_candidate)"],
                    "source_name": source,
                    "source_embedding": source_embedding,
                    "user_id": user_id,
                }
                if agent_id:
                    params["agent_id"] = agent_id

            elif source_node_search_result and destination_node_search_result:
                cypher = f"""
                MATCH (source)
                WHERE elementId(source) = $source_id
                SET source.mentions = coalesce(source.mentions, 0) + 1
                WITH source
                MATCH (destination)
                WHERE elementId(destination) = $destination_id
                SET destination.mentions = coalesce(destination.mentions, 0) + 1
                MERGE (source)-[r:{relationship}]->(destination)
                ON CREATE SET 
                    r.created_at = timestamp(),
                    r.updated_at = timestamp(),
                    r.mentions = 1
                ON MATCH SET r.mentions = coalesce(r.mentions, 0) + 1
                RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                """

                params = {
                    "source_id": source_node_search_result[0]["elementId(source_candidate)"],
                    "destination_id": destination_node_search_result[0]["elementId(destination_candidate)"],
                    "user_id": user_id,
                }
                if agent_id:
                    params["agent_id"] = agent_id

            else:
                # Build dynamic MERGE props for both source and destination
                source_props = ["name: $source_name", "user_id: $user_id"]
                dest_props = ["name: $dest_name", "user_id: $user_id"]
                if agent_id:
                    source_props.append("agent_id: $agent_id")
                    dest_props.append("agent_id: $agent_id")
                source_props_str = ", ".join(source_props)
                dest_props_str = ", ".join(dest_props)

                # Build temporal property strings for TCMGM
                temporal_props = []
                if timestamp:
                    temporal_props.append(f"source.timestamp = $timestamp, destination.timestamp = $timestamp")
                if confidence:
                    temporal_props.append(f"source.confidence = $confidence, destination.confidence = $confidence")
                if modality:
                    temporal_props.append(f"source.modality = $modality, destination.modality = $modality")
                if image_hash:
                    temporal_props.append(f"source.image_hash = $image_hash, destination.image_hash = $image_hash")
                if audio_hash:
                    temporal_props.append(f"source.audio_hash = $audio_hash, destination.audio_hash = $audio_hash")
                
                temporal_set = ", " + ", ".join(temporal_props) if temporal_props else ""

                cypher = f"""
                MERGE (source {source_label} {{{source_props_str}}})
                ON CREATE SET source.created = timestamp(),
                            source.mentions = 1
                            {source_extra_set}
                            {temporal_set.replace("destination.", "source.")}
                ON MATCH SET source.mentions = coalesce(source.mentions, 0) + 1
                WITH source
                CALL db.create.setNodeVectorProperty(source, 'embedding', $source_embedding)
                WITH source
                MERGE (destination {destination_label} {{{dest_props_str}}})
                ON CREATE SET destination.created = timestamp(),
                            destination.mentions = 1
                            {destination_extra_set}
                            {temporal_set.replace("source.", "destination.")}
                ON MATCH SET destination.mentions = coalesce(destination.mentions, 0) + 1
                WITH source, destination
                CALL db.create.setNodeVectorProperty(destination, 'embedding', $dest_embedding)
                WITH source, destination
                MERGE (source)-[rel:{relationship}]->(destination)
                ON CREATE SET rel.created = timestamp(), rel.mentions = 1
                ON MATCH SET rel.mentions = coalesce(rel.mentions, 0) + 1
                RETURN source.name AS source, type(rel) AS relationship, destination.name AS target
                """

                params = {
                    "source_name": source,
                    "dest_name": destination,
                    "source_embedding": source_embedding,
                    "dest_embedding": dest_embedding,
                    "user_id": user_id,
                }
                if agent_id:
                    params["agent_id"] = agent_id
                
                # Add temporal parameters
                if timestamp:
                    params["timestamp"] = timestamp
                if confidence:
                    params["confidence"] = confidence
                if modality:
                    params["modality"] = modality
                if image_hash:
                    params["image_hash"] = image_hash
                if audio_hash:
                    params["audio_hash"] = audio_hash
            result = self.graph.query(cypher, params=params)
            results.append(result)
        return results

    def _remove_spaces_from_entities(self, entity_list):
        for item in entity_list:
            item["source"] = item["source"].lower().replace(" ", "_")
            item["relationship"] = item["relationship"].lower().replace(" ", "_")
            item["destination"] = item["destination"].lower().replace(" ", "_")
        return entity_list

    def _search_source_node(self, source_embedding, filters, threshold=0.9):
        agent_filter = ""
        if filters.get("agent_id"):
            agent_filter = "AND source_candidate.agent_id = $agent_id"

        cypher = f"""
            MATCH (source_candidate {self.node_label})
            WHERE source_candidate.embedding IS NOT NULL 
            AND source_candidate.user_id = $user_id
            {agent_filter}

            WITH source_candidate,
            round(2 * vector.similarity.cosine(source_candidate.embedding, $source_embedding) - 1, 4) AS source_similarity // denormalize for backward compatibility
            WHERE source_similarity >= $threshold

            WITH source_candidate, source_similarity
            ORDER BY source_similarity DESC
            LIMIT 1

            RETURN elementId(source_candidate)
            """

        params = {
            "source_embedding": source_embedding,
            "user_id": filters["user_id"],
            "threshold": threshold,
        }
        if filters.get("agent_id"):
            params["agent_id"] = filters["agent_id"]

        result = self.graph.query(cypher, params=params)
        return result

    def _search_destination_node(self, destination_embedding, filters, threshold=0.9):
        agent_filter = ""
        if filters.get("agent_id"):
            agent_filter = "AND destination_candidate.agent_id = $agent_id"

        cypher = f"""
            MATCH (destination_candidate {self.node_label})
            WHERE destination_candidate.embedding IS NOT NULL 
            AND destination_candidate.user_id = $user_id
            {agent_filter}

            WITH destination_candidate,
            round(2 * vector.similarity.cosine(destination_candidate.embedding, $destination_embedding) - 1, 4) AS destination_similarity // denormalize for backward compatibility

            WHERE destination_similarity >= $threshold

            WITH destination_candidate, destination_similarity
            ORDER BY destination_similarity DESC
            LIMIT 1

            RETURN elementId(destination_candidate)
            """

        params = {
            "destination_embedding": destination_embedding,
            "user_id": filters["user_id"],
            "threshold": threshold,
        }
        if filters.get("agent_id"):
            params["agent_id"] = filters["agent_id"]

        result = self.graph.query(cypher, params=params)
        return result

    def add_causal_link(self, causal_link, filters: dict):
        """
        Add a causal relationship to the graph (TCMGM Phase 2).
        
        Args:
            causal_link: CausalLink object or dict with cause_id, effect_id, causal_type, confidence, evidence, timestamp
            filters: Query filters (user_id, agent_id, etc.)
        """
        # Handle both CausalLink objects and dicts
        if hasattr(causal_link, 'cause_id'):
            cause_id = causal_link.cause_id
            effect_id = causal_link.effect_id
            causal_type = causal_link.causal_type
            confidence = causal_link.confidence
            evidence = causal_link.evidence or ""
            timestamp = causal_link.timestamp.isoformat()
        else:
            cause_id = causal_link.get("cause_id")
            effect_id = causal_link.get("effect_id")
            causal_type = causal_link.get("causal_type")
            confidence = causal_link.get("confidence", 0.7)
            evidence = causal_link.get("evidence", "")
            timestamp = causal_link.get("timestamp", "")
            if hasattr(timestamp, 'isoformat'):
                timestamp = timestamp.isoformat()
        
        # Build agent filter
        agent_filter = ""
        params = {
            "cause_id": cause_id,
            "effect_id": effect_id,
            "causal_type": causal_type,
            "confidence": confidence,
            "evidence": evidence,
            "timestamp": timestamp,
            "user_id": filters.get("user_id")
        }
        
        if filters.get("agent_id"):
            agent_filter = "AND cause.agent_id = $agent_id AND effect.agent_id = $agent_id"
            params["agent_id"] = filters.get("agent_id")
        
        # Create causal relationship in Neo4j
        cypher = f"""
        MATCH (cause {self.node_label} {{name: $cause_id, user_id: $user_id}})
        MATCH (effect {self.node_label} {{name: $effect_id, user_id: $user_id}})
        WHERE 1=1 {agent_filter}
        MERGE (cause)-[r:CAUSAL {{
            type: $causal_type,
            confidence: $confidence,
            evidence: $evidence,
            timestamp: $timestamp
        }}]->(effect)
        RETURN r
        """
        
        try:
            result = self.graph.query(cypher, params=params)
            logger.info(f"Added causal link: {cause_id} → {effect_id} ({causal_type})")
            return result
        except Exception as e:
            logger.error(f"Failed to add causal link: {e}", exc_info=True)
            return None

    def add_causal_links_batch(self, causal_links, filters: dict):
        """Phase PT2 — bulk-insert N causal links in one Cypher round-trip.

        The legacy ``add_causal_link`` issues one MERGE per link
        (= N network round-trips). Causal extraction commonly yields
        5-20 links per event batch, so the round-trip overhead is the
        single biggest write-side cost. This method collapses every
        link into a single ``UNWIND`` MERGE so the wire-time cost is
        one round-trip + one transaction commit regardless of N.

        Failure semantics: the entire batch is one transaction. Any
        link that violates a constraint or references a missing
        cause/effect node aborts the whole batch — ``None`` is
        returned. Callers that want per-link recovery should fall
        back to ``add_causal_link`` (legacy path stays unchanged).

        Args:
            causal_links: iterable of ``CausalLink`` objects or dicts.
            filters: query filters (must contain ``user_id``).

        Returns:
            Cypher result rows on success, ``None`` on batch failure
            or empty input.
        """
        if not causal_links:
            return None

        rows = []
        for cl in causal_links:
            if hasattr(cl, "cause_id"):
                rows.append({
                    "cause_id": cl.cause_id,
                    "effect_id": cl.effect_id,
                    "causal_type": cl.causal_type,
                    "confidence": cl.confidence,
                    "evidence": cl.evidence or "",
                    "timestamp": cl.timestamp.isoformat() if cl.timestamp else "",
                })
            else:
                ts = cl.get("timestamp", "")
                if hasattr(ts, "isoformat"):
                    ts = ts.isoformat()
                rows.append({
                    "cause_id": cl.get("cause_id"),
                    "effect_id": cl.get("effect_id"),
                    "causal_type": cl.get("causal_type"),
                    "confidence": cl.get("confidence", 0.7),
                    "evidence": cl.get("evidence", "") or "",
                    "timestamp": ts,
                })

        params = {"rows": rows, "user_id": filters.get("user_id")}
        agent_filter = ""
        if filters.get("agent_id"):
            agent_filter = "AND cause.agent_id = $agent_id AND effect.agent_id = $agent_id"
            params["agent_id"] = filters.get("agent_id")

        cypher = f"""
        UNWIND $rows AS row
        MATCH (cause {self.node_label} {{name: row.cause_id, user_id: $user_id}})
        MATCH (effect {self.node_label} {{name: row.effect_id, user_id: $user_id}})
        WHERE 1=1 {agent_filter}
        MERGE (cause)-[r:CAUSAL {{
            type: row.causal_type,
            confidence: row.confidence,
            evidence: row.evidence,
            timestamp: row.timestamp
        }}]->(effect)
        RETURN count(r) AS created
        """
        try:
            result = self.graph.query(cypher, params=params)
            logger.info(
                "Bulk-inserted %d causal links via UNWIND (PT2)", len(rows)
            )
            return result
        except Exception as e:
            logger.error(
                "Bulk causal insert failed (%d links): %s", len(rows), e,
                exc_info=True,
            )
            return None

    def add_multimodal_event(
        self,
        content,
        modality: str,
        filters: dict,
        timestamp=None,
        confidence=1.0,
        metadata=None
    ):
        """
        Add multimodal event to graph (TCMGM Phase 3).
        
        Supports text, image, audio, and other modalities with cross-modal linking.
        
        Args:
            content: Content in any modality (str, bytes, PIL.Image, etc.)
            modality: Modality type (text, image, audio, video, etc.)
            filters: Query filters (user_id, agent_id, etc.)
            timestamp: Event timestamp (default: now)
            confidence: Confidence score (0-1)
            metadata: Additional metadata dict
        
        Returns:
            Result from graph add operation
            
        Example:
            >>> from PIL import Image
            >>> img = Image.open("photo.jpg")
            >>> memory_graph.add_multimodal_event(
            ...     content=img,
            ...     modality="image",
            ...     filters={"user_id": "user_123"},
            ...     metadata={"source": "camera"}
            ... )
        """
        from outhad_contextkit.memory.temporal.multimodal import (
            MultimodalContent,
            MultimodalEmbedder
        )
        from datetime import datetime
        
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        # Create multimodal content wrapper
        mm_content = MultimodalContent(content, modality, metadata)
        content_hash = mm_content.compute_hash()
        
        logger.info(f"Adding multimodal event: modality={modality}, hash={content_hash[:8]}")
        
        # Generate embedding
        embedder = MultimodalEmbedder(self.embedding_model)
        try:
            embedding = embedder.embed_multimodal(mm_content)
            mm_content.set_embedding(embedding)
        except Exception as e:
            logger.error(f"Failed to generate embedding for {modality}: {e}")
            # Use zero vector as fallback
            embedding = [0.0] * 768
        
        # Prepare data for storage
        if modality == "text":
            data_to_store = content if isinstance(content, str) else str(content)
        else:
            # Store modality prefix + hash reference for non-text
            data_to_store = f"{modality}:{content_hash[:16]}"
        
        # Prepare metadata for storage
        storage_metadata = {
            **(metadata or {}),
            "content_hash": content_hash,
            "content_size": mm_content.get_size(),
            "embedding_dim": len(embedding)
        }
        
        # Add to graph with multimodal attributes
        result = self.add(
            data=data_to_store,
            filters=filters,
            timestamp=timestamp,
            confidence=confidence,
            modality=modality,
            image_hash=content_hash if modality == "image" else None,
            audio_hash=content_hash if modality == "audio" else None,
            metadata=storage_metadata
        )
        
        logger.info(f"✅ Added {modality} event with hash {content_hash[:8]}")
        return result
    
    def search_multimodal(
        self,
        query_content,
        query_modality: str,
        filters: dict,
        top_k: int = 5,
        modality_filter: str = None
    ):
        """
        Search across modalities (TCMGM Phase 3).
        
        Find events across different modalities using semantic similarity.
        For example: search for images using text queries.
        
        Args:
            query_content: Query content (text, image, etc.)
            query_modality: Modality of query
            filters: Query filters (user_id, agent_id, etc.)
            top_k: Number of results to return
            modality_filter: Optional filter for result modality
        
        Returns:
            List of matching events across modalities
            
        Example:
            >>> results = memory_graph.search_multimodal(
            ...     query_content="red sunset",
            ...     query_modality="text",
            ...     filters={"user_id": "user_123"},
            ...     modality_filter="image"  # Find images matching text
            ... )
        """
        from outhad_contextkit.memory.temporal.multimodal import (
            MultimodalContent,
            MultimodalEmbedder
        )
        from outhad_contextkit.memory.temporal.cross_modal import cross_modal_search
        
        # Generate query embedding
        query_mm = MultimodalContent(query_content, query_modality)
        embedder = MultimodalEmbedder(self.embedding_model)
        
        try:
            query_embedding = embedder.embed_multimodal(query_mm)
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            return []
        
        # Search graph for events with embeddings
        user_id = filters.get("user_id")
        agent_filter = ""
        params = {"user_id": user_id}
        
        if filters.get("agent_id"):
            agent_filter = "AND n.agent_id = $agent_id"
            params["agent_id"] = filters.get("agent_id")
        
        if modality_filter:
            agent_filter += " AND n.modality = $modality"
            params["modality"] = modality_filter
        
        # Query for nodes with embeddings
        cypher = f"""
        MATCH (n {self.node_label})
        WHERE n.user_id = $user_id {agent_filter}
        AND n.embedding IS NOT NULL
        RETURN n.name AS name,
               n.modality AS modality,
               n.embedding AS embedding,
               n.timestamp AS timestamp,
               n.confidence AS confidence,
               n.metadata AS metadata
        LIMIT 100
        """
        
        try:
            results = self.graph.query(cypher, params=params)
            
            # Prepare candidates for cross-modal search
            candidates = []
            for record in results:
                candidates.append({
                    'id': record.get('name'),
                    'content': record.get('name'),
                    'modality': record.get('modality', 'text'),
                    'embedding': record.get('embedding', []),
                    'metadata': record.get('metadata', {}),
                    'timestamp': record.get('timestamp'),
                    'confidence': record.get('confidence', 1.0)
                })
            
            # Perform cross-modal search
            matches = cross_modal_search(
                query_embedding=query_embedding,
                candidate_embeddings=candidates,
                modality_filter=modality_filter,
                top_k=top_k
            )
            
            logger.info(f"Cross-modal search: found {len(matches)} matches")
            return matches
            
        except Exception as e:
            logger.error(f"Multimodal search failed: {e}", exc_info=True)
            return []

    # Reset is not defined in base.py
    def reset(self):
        """Reset the graph by clearing all nodes and relationships."""
        logger.warning("Clearing graph...")
        cypher_query = """
        MATCH (n) DETACH DELETE n
        """
        return self.graph.query(cypher_query)
