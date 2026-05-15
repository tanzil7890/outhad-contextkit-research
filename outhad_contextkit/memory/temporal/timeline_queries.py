"""Timeline-specific query utilities."""
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta

from outhad_contextkit.memory.temporal.types import TimeWindow

logger = logging.getLogger(__name__)


class TimelineQueries:
    """Helper class for timeline queries."""
    
    def __init__(self, timeline_builder):
        """
        Initialize timeline queries.
        
        Args:
            timeline_builder: TimelineBuilder instance
        """
        self.timeline = timeline_builder
    
    def what_happened_after(
        self,
        user_id: str,
        event_description: str,
        limit: int = 10
    ) -> List[Dict]:
        """
        Answer 'What happened after X?' queries.
        
        Args:
            user_id: User identifier
            event_description: Description of the reference event
            limit: Maximum events to return
        
        Returns:
            List of events that happened after the reference event
        """
        # First find the reference event
        reference_event = self.find_event_by_description(user_id, event_description)
        
        if not reference_event:
            logger.warning(f"Could not find reference event: {event_description}")
            return []
        
        # Get subsequent events
        # Try to get event ID from different possible fields
        event_id = reference_event.get('id') or reference_event.get('name')
        if not event_id:
            logger.warning(f"Reference event found but has no id: {reference_event}")
            return []
        
        return self.timeline.get_events_after(
            user_id,
            event_id,
            limit=limit
        )
    
    def what_happened_before(
        self,
        user_id: str,
        event_description: str,
        limit: int = 10
    ) -> List[Dict]:
        """
        Answer 'What happened before X?' queries.
        
        Args:
            user_id: User identifier
            event_description: Description of the reference event
            limit: Maximum events to return
        
        Returns:
            List of events that happened before the reference event
        """
        # First find the reference event
        reference_event = self.find_event_by_description(user_id, event_description)
        
        if not reference_event:
            logger.warning(f"Could not find reference event: {event_description}")
            return []
        
        # Get preceding events
        # Try to get event ID from different possible fields
        event_id = reference_event.get('id') or reference_event.get('name')
        if not event_id:
            logger.warning(f"Reference event found but has no id: {reference_event}")
            return []
        
        return self.timeline.get_events_before(
            user_id,
            event_id,
            limit=limit
        )
    
    def events_on_date(self, user_id: str, date: datetime) -> List[Dict]:
        """
        Get all events on a specific date.
        
        Args:
            user_id: User identifier
            date: Target date
        
        Returns:
            List of events on that date
        """
        start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end = date.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        return self.timeline.get_events_between(user_id, start, end)
    
    def events_this_week(self, user_id: str) -> List[Dict]:
        """
        Get events from the current week.
        
        Args:
            user_id: User identifier
        
        Returns:
            List of events from this week
        """
        now = datetime.utcnow()
        week_start = now - timedelta(days=now.weekday())
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        
        return self.timeline.get_events_between(user_id, week_start, now)
    
    def events_last_n_days(self, user_id: str, n_days: int) -> List[Dict]:
        """
        Get events from the last N days.
        
        Args:
            user_id: User identifier
            n_days: Number of days to look back
        
        Returns:
            List of events from the last N days
        """
        now = datetime.utcnow()
        start = now - timedelta(days=n_days)
        
        return self.timeline.get_events_between(user_id, start, now)
    
    def events_between_dates(
        self,
        user_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict]:
        """
        Get events between two dates.
        
        Args:
            user_id: User identifier
            start_date: Start date
            end_date: End date
        
        Returns:
            List of events between the dates
        """
        return self.timeline.get_events_between(user_id, start_date, end_date)
    
    def find_event_by_description(
        self,
        user_id: str,
        description: str,
        time_window: Optional[TimeWindow] = None
    ) -> Optional[Dict]:
        """
        Find event matching description.
        
        Args:
            user_id: User identifier
            description: Event description to search for
            time_window: Optional time window to narrow search
        
        Returns:
            First matching event or None
        """
        events = self.timeline.get_timeline(user_id, time_window)
        
        # Simple text matching (can be enhanced with embeddings)
        description_lower = description.lower()
        for event in events:
            event_content = event.get('content', event.get('name', '')).lower()
            if description_lower in event_content:
                return event
        
        return None
    
    def find_events_by_modality(
        self,
        user_id: str,
        modality: str,
        time_window: Optional[TimeWindow] = None,
        limit: int = 50
    ) -> List[Dict]:
        """
        Find events by modality type.
        
        Args:
            user_id: User identifier
            modality: Modality type (text, image, audio, etc.)
            time_window: Optional time window filter
            limit: Maximum events to return
        
        Returns:
            List of events with the specified modality
        """
        all_events = self.timeline.get_timeline(user_id, time_window, limit=limit)
        
        return [e for e in all_events if e.get('modality') == modality]
    
    def get_recent_events(self, user_id: str, limit: int = 10) -> List[Dict]:
        """
        Get the most recent events.
        
        Args:
            user_id: User identifier
            limit: Maximum events to return
        
        Returns:
            List of recent events
        """
        return self.timeline.get_timeline(user_id, time_window=None, limit=limit)
    
    def search_timeline(
        self,
        user_id: str,
        query: str,
        time_window: Optional[TimeWindow] = None,
        limit: int = 20
    ) -> List[Dict]:
        """
        Search timeline with text query.
        
        Args:
            user_id: User identifier
            query: Search query
            time_window: Optional time window filter
            limit: Maximum events to return
        
        Returns:
            List of matching events
        """
        events = self.timeline.get_timeline(user_id, time_window, limit=limit * 2)
        
        # Simple text search (can be enhanced with semantic search)
        query_lower = query.lower()
        matches = []
        
        for event in events:
            content = event.get('content', event.get('name', '')).lower()
            if query_lower in content:
                matches.append(event)
                if len(matches) >= limit:
                    break
        
        return matches
    
    def get_timeline_summary_for_period(
        self,
        user_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> str:
        """
        Get natural language summary for a time period.
        
        Args:
            user_id: User identifier
            start_date: Period start
            end_date: Period end
        
        Returns:
            Timeline summary string
        """
        time_window = TimeWindow(start=start_date, end=end_date)
        return self.timeline.summarize_timeline(user_id, time_window)

