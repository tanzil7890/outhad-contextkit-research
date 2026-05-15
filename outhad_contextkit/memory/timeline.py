"""Timeline helper for building temporal event graphs."""
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from collections import defaultdict

from outhad_contextkit.memory.temporal.types import TemporalEvent, TimeWindow
from outhad_contextkit.memory.temporal.causal_extractor import CausalExtractor

logger = logging.getLogger(__name__)


class TimelineBuilder:
    """Builds temporal event timelines from transcripts."""
    
    def __init__(self, llm, graph):
        """
        Initialize timeline builder.
        
        Args:
            llm: LLM instance for event extraction
            graph: Graph memory instance
        """
        self.llm = llm
        self.graph = graph
        self.causal_extractor = CausalExtractor(llm)
    
    def build_from_transcript(
        self,
        transcript: str,
        user_id: str,
        session_start: datetime = None
    ) -> Dict:
        """
        Build timeline from conversation transcript.
        
        Args:
            transcript: Conversation text
            user_id: User identifier
            session_start: Session start time
        
        Returns:
            Dict with events and causal links
        """
        if session_start is None:
            session_start = datetime.utcnow()
        
        # Extract events from transcript
        events = self._extract_events(transcript, session_start, user_id)
        
        # Extract causal relationships
        causal_links = self.causal_extractor.extract_causal_links(events)
        
        # Store in graph
        self._store_timeline(events, causal_links, user_id)
        
        return {
            "events": events,
            "causal_links": causal_links,
            "timeline_span": {
                "start": events[0]['timestamp'] if events else None,
                "end": events[-1]['timestamp'] if events else None
            }
        }
    
    def _extract_events(
        self,
        transcript: str,
        session_start: datetime,
        user_id: str = "unknown"
    ) -> List[Dict]:
        """Extract individual events from transcript."""
        from outhad_contextkit.memory.utils import get_fact_retrieval_messages
        
        # Use existing fact extraction
        messages = get_fact_retrieval_messages(transcript)
        
        try:
            response = self.llm.generate_response(
                messages=[
                    {"role": "system", "content": messages[0]},
                    {"role": "user", "content": messages[1]}
                ],
                response_format={"type": "json_object"}
            )
            
            import json
            facts = json.loads(response).get("facts", [])
            
            # Convert facts to temporal events
            events = []
            for i, fact in enumerate(facts):
                # Estimate timestamp based on position in transcript
                time_offset = timedelta(minutes=i * 2)  # Assume 2 min per fact
                
                event = {
                    "id": f"event_{i}_{user_id}_{session_start.isoformat()}",
                    "content": fact,
                    "timestamp": (session_start + time_offset).isoformat(),
                    "confidence": 0.85,
                    "modality": "text"
                }
                events.append(event)
            
            logger.info(f"Extracted {len(events)} events from transcript")
            return events
        
        except Exception as e:
            logger.error(f"Event extraction failed: {e}")
            return []
    
    def _store_timeline(
        self,
        events: List[Dict],
        causal_links: List,
        user_id: str
    ):
        """Store timeline in graph."""
        filters = {"user_id": user_id}
        
        # Store events
        for event in events:
            try:
                self.graph.add(
                    data=event['content'],
                    filters=filters,
                    timestamp=datetime.fromisoformat(event['timestamp']),
                    confidence=event['confidence'],
                    modality=event['modality']
                )
            except Exception as e:
                logger.warning(f"Failed to store event {event['id']}: {e}")
        
        # Store causal links
        for link in causal_links:
            try:
                self.graph.add_causal_link(link, filters)
            except Exception as e:
                logger.warning(f"Failed to store causal link: {e}")
        
        logger.info(f"Stored {len(events)} events and {len(causal_links)} causal links")
    
    def get_timeline(
        self,
        user_id: str,
        time_window: TimeWindow = None,
        limit: int = 50
    ) -> List[Dict]:
        """
        Retrieve timeline for user.
        
        Args:
            user_id: User identifier
            time_window: Optional time window filter
            limit: Maximum events to return
        
        Returns:
            List of events in chronological order
        """
        try:
            # Build Cypher query
            cypher = """
            MATCH (n {user_id: $user_id})
            WHERE n.timestamp IS NOT NULL
            """
            
            params = {"user_id": user_id, "limit": limit}
            
            if time_window and time_window.start:
                cypher += " AND n.timestamp >= $start"
                params["start"] = time_window.start.isoformat()
            
            if time_window and time_window.end:
                cypher += " AND n.timestamp <= $end"
                params["end"] = time_window.end.isoformat()
            
            cypher += """
            RETURN n
            ORDER BY n.timestamp ASC
            LIMIT $limit
            """
            
            results = self.graph.graph.query(cypher, params=params)
            
            return [dict(r["n"]) for r in results]
        
        except Exception as e:
            logger.error(f"Timeline retrieval failed: {e}")
            return []
    
    def get_events_between(
        self,
        user_id: str,
        start: datetime,
        end: datetime
    ) -> List[Dict]:
        """Get events within time range."""
        window = TimeWindow(start=start, end=end)
        return self.get_timeline(user_id, window)
    
    def get_events_after(
        self,
        user_id: str,
        reference_event_id: str,
        limit: int = 10
    ) -> List[Dict]:
        """Get events that occurred after a reference event."""
        try:
            cypher = """
            MATCH (ref {id: $ref_id, user_id: $user_id})
            MATCH (n {user_id: $user_id})
            WHERE n.timestamp > ref.timestamp
            RETURN n
            ORDER BY n.timestamp ASC
            LIMIT $limit
            """
            
            params = {
                "ref_id": reference_event_id,
                "user_id": user_id,
                "limit": limit
            }
            
            results = self.graph.graph.query(cypher, params=params)
            return [dict(r["n"]) for r in results]
        
        except Exception as e:
            logger.error(f"Events after retrieval failed: {e}")
            return []
    
    def get_events_before(
        self,
        user_id: str,
        reference_event_id: str,
        limit: int = 10
    ) -> List[Dict]:
        """Get events that occurred before a reference event."""
        try:
            cypher = """
            MATCH (ref {id: $ref_id, user_id: $user_id})
            MATCH (n {user_id: $user_id})
            WHERE n.timestamp < ref.timestamp
            RETURN n
            ORDER BY n.timestamp DESC
            LIMIT $limit
            """
            
            params = {
                "ref_id": reference_event_id,
                "user_id": user_id,
                "limit": limit
            }
            
            results = self.graph.graph.query(cypher, params=params)
            # Reverse to maintain chronological order
            return [dict(r["n"]) for r in reversed(results)]
        
        except Exception as e:
            logger.error(f"Events before retrieval failed: {e}")
            return []
    
    def summarize_timeline(
        self,
        user_id: str,
        time_window: TimeWindow = None
    ) -> str:
        """Generate natural language summary of timeline."""
        events = self.get_timeline(user_id, time_window)
        
        if not events:
            return "No events in timeline."
        
        summary_prompt = f"""Summarize this timeline of events:

Events:
{chr(10).join(f"{i+1}. {e.get('content', e.get('name', 'Unknown event'))} (at {e.get('timestamp', 'unknown time')})" for i, e in enumerate(events))}

Provide a concise narrative summary."""
        
        try:
            response = self.llm.generate_response(
                messages=[{"role": "user", "content": summary_prompt}]
            )
            return response
        except Exception as e:
            logger.error(f"Timeline summarization failed: {e}")
            return "Unable to generate summary."
    
    def get_timeline_stats(self, user_id: str, time_window: TimeWindow = None) -> Dict:
        """Get statistics about the timeline."""
        events = self.get_timeline(user_id, time_window)
        
        if not events:
            return {
                "total_events": 0,
                "modalities": {},
                "time_span": None,
                "avg_confidence": 0.0
            }
        
        # Count modalities
        modalities = defaultdict(int)
        for event in events:
            modality = event.get('modality', 'unknown')
            modalities[modality] += 1
        
        # Get time span
        timestamps = [e.get('timestamp') for e in events if e.get('timestamp')]
        time_span = None
        if timestamps:
            time_span = {
                "start": min(timestamps),
                "end": max(timestamps)
            }
        
        return {
            "total_events": len(events),
            "modalities": dict(modalities),
            "time_span": time_span,
            "avg_confidence": sum(e.get('confidence', 0) for e in events) / len(events) if events else 0
        }

