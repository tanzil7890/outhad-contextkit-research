"""Cross-modal retrieval utilities."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


#  calibrated per-(query_modality, candidate_modality)
# similarity thresholds. CLIP image↔text cosines centre around 0.20–
# 0.30, while text↔text in the same model space centre around 0.50+.
# A single global threshold (0.0 today) admits noise on text↔text and
# misses true positives on image↔text. The map below was tuned on a
# CLIP-vit-base-patch32 held-out set; deploys using SigLIP / OpenAI
# embeddings can override via :data:`MODALITY_THRESHOLDS`.
MODALITY_THRESHOLDS: Dict[Tuple[str, str], float] = {
    ("text", "text"): 0.55,
    ("text", "image"): 0.22,
    ("image", "text"): 0.22,
    ("text", "audio"): 0.18,
    ("audio", "text"): 0.18,
    ("image", "image"): 0.50,
    ("audio", "audio"): 0.45,
    ("image", "audio"): 0.15,
    ("audio", "image"): 0.15,
}


def _threshold_for(
    query_modality: Optional[str],
    candidate_modality: Optional[str],
    floor: float,
) -> float:
    """Resolve the calibrated threshold for the given modality pair.

    Falls back to ``floor`` (caller-supplied global ``min_similarity``)
    when:
      * Either side's modality is missing.
      * The pair is not in the calibration map.
    """
    if not query_modality or not candidate_modality:
        return floor
    calibrated = MODALITY_THRESHOLDS.get(
        (query_modality, candidate_modality)
    )
    if calibrated is None:
        return floor
    return max(floor, calibrated)


def cross_modal_search(
    query_embedding: List[float],
    candidate_embeddings: List[Dict],
    modality_filter: Optional[str] = None,
    top_k: int = 5,
    min_similarity: float = 0.0,
    *,
    query_modality: Optional[str] = None,
    use_calibrated_thresholds: bool = True,
    time_window: Optional[Tuple[datetime, datetime]] = None,
    prefilter_max_candidates: Optional[int] = None,
) -> List[Dict]:
    """
    Search across modalities using embedding similarity.

    
    * **prefilter** : when ``time_window`` is supplied, candidates
      outside the window are skipped *before* the cosine loop. When
      ``prefilter_max_candidates`` is set, the candidate list is capped
      after modality + time filters but before similarity scoring.
    * **calibrated thresholds** (A4): when
      ``use_calibrated_thresholds=True`` (default) and ``query_modality``
      is supplied, each candidate is judged against the per-pair
      threshold from :data:`MODALITY_THRESHOLDS` instead of a single
      global ``min_similarity`` floor.

    Args:
        query_embedding: Query embedding vector.
        candidate_embeddings: List of dicts with ``embedding``,
            ``modality``, ``content`` keys.
        modality_filter: Optional modality filter (e.g. ``"image"``).
        top_k: Number of results to return.
        min_similarity: Global similarity floor; combined with the
            calibrated per-pair threshold via ``max(...)``.
        query_modality: Modality of the query (``"text"`` /
            ``"image"`` / ``"audio"``). Required for calibrated
            thresholds; ignored otherwise.
        use_calibrated_thresholds: When True, apply the per-pair
            threshold map from :data:`MODALITY_THRESHOLDS`.
        time_window: Optional ``(start, end)`` datetime tuple. Only
            candidates whose ``timestamp`` falls inside the window
            survive the prefilter.
        prefilter_max_candidates: Optional cap on the number of
            candidates that reach the cosine loop. Useful on large
            stores when running cross-modal exploration as a
            best-effort augmentation.

    Returns:
        Top-k most similar results across modalities, sorted by
        similarity.
    """
    if not candidate_embeddings:
        logger.warning("No candidate embeddings provided")
        return []

    query_vec = np.array(query_embedding)
    if np.linalg.norm(query_vec) == 0:
        logger.warning("Query embedding has zero norm")
        return []

    #  prefilter (modality + time + budget cap).
    pool: List[Dict] = []
    initial = len(candidate_embeddings)
    for candidate in candidate_embeddings:
        if modality_filter and candidate.get("modality") != modality_filter:
            continue
        if time_window is not None:
            ts = candidate.get("timestamp")
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts)
                except (TypeError, ValueError):
                    ts = None
            if ts is not None:
                start, end = time_window
                if (start and ts < start) or (end and ts > end):
                    continue
        pool.append(candidate)
    if prefilter_max_candidates is not None and prefilter_max_candidates > 0:
        pool = pool[: int(prefilter_max_candidates)]
    if len(pool) < initial:
        logger.debug(
            "Cross-modal prefilter: %d → %d candidates", initial, len(pool)
        )

    scores: List[Dict] = []
    for candidate in pool:
        candidate_vec = np.array(candidate.get("embedding") or [])
        candidate_norm = np.linalg.norm(candidate_vec)
        if candidate_norm == 0:
            logger.debug(
                "Skipping candidate with zero norm: %s",
                candidate.get("content", "unknown"),
            )
            continue
        similarity = float(
            np.dot(query_vec, candidate_vec)
            / (np.linalg.norm(query_vec) * candidate_norm)
        )

        #  calibrated per-pair threshold.
        threshold = min_similarity
        if use_calibrated_thresholds:
            threshold = _threshold_for(
                query_modality,
                candidate.get("modality"),
                floor=min_similarity,
            )
        if similarity < threshold:
            continue

        scores.append({
            "content": candidate.get("content", ""),
            "modality": candidate.get("modality", "unknown"),
            "similarity": similarity,
            "threshold_applied": float(threshold),
            "metadata": candidate.get("metadata", {}),
            "id": candidate.get("id"),
        })

    scores.sort(key=lambda x: x["similarity"], reverse=True)
    logger.info(
        "Cross-modal search: %d results (filtered from %d candidates)",
        len(scores),
        initial,
    )
    return scores[:top_k]


def find_image_mentions_in_text(
    image_hash: str,
    text_events: List[Dict],
    window_minutes: int = 60,
    image_timestamp: datetime = None
) -> List[Dict]:
    """
    Find text events that mention or reference an image.
    
    This enables finding conversations about images, descriptions,
    or any text that references visual content.
    
    Args:
        image_hash: Hash of the image to find mentions of
        text_events: List of text event dicts with 'content' and optionally 'timestamp'
        window_minutes: Time window to search (minutes) - only used if timestamps available
        image_timestamp: Timestamp of the image (for temporal filtering)
    
    Returns:
        List of dicts containing matched events with confidence scores
        
    Example:
        >>> text_events = [
        ...     {"content": "Look at this screenshot", "timestamp": "..."},
        ...     {"content": "The image shows a bug", "timestamp": "..."}
        ... ]
        >>> mentions = find_image_mentions_in_text("abc123", text_events)
    """
    mentions = []
    
    # Define image reference keywords
    image_keywords = [
        "image", "picture", "photo", "screenshot", "showed", "attached",
        "look at", "see this", "visual", "diagram", "chart", "graph",
        "illustration", "figure", "snapshot"
    ]
    
    for event in text_events:
        content_lower = event.get('content', '').lower()
        
        # Check for temporal proximity if timestamps available
        if image_timestamp and 'timestamp' in event:
            try:
                event_time = datetime.fromisoformat(event['timestamp']) if isinstance(event['timestamp'], str) else event['timestamp']
                time_diff = abs((event_time - image_timestamp).total_seconds() / 60)  # minutes
                
                if time_diff > window_minutes:
                    continue  # Outside time window
                    
                # Boost confidence for temporally close events
                temporal_boost = 1.0 - (time_diff / window_minutes) * 0.3
            except (ValueError, TypeError):
                temporal_boost = 1.0
        else:
            temporal_boost = 1.0
        
        # Check for image reference keywords
        keyword_matches = [kw for kw in image_keywords if kw in content_lower]
        
        if keyword_matches:
            confidence = 0.6 * temporal_boost + 0.1 * min(len(keyword_matches), 3)
            
            mentions.append({
                'event': event,
                'confidence': min(confidence, 1.0),
                'reason': f"Contains keywords: {', '.join(keyword_matches)}",
                'matched_keywords': keyword_matches
            })
        
        # Check for direct hash reference (exact match)
        if image_hash and image_hash[:8] in event.get('content', ''):
            mentions.append({
                'event': event,
                'confidence': 0.95,
                'reason': 'Direct hash reference',
                'matched_keywords': []
            })
    
    # Sort by confidence
    mentions.sort(key=lambda x: x['confidence'], reverse=True)
    
    logger.info(f"Found {len(mentions)} potential image mentions in {len(text_events)} text events")
    
    return mentions


def find_audio_mentions_in_text(
    audio_hash: str,
    text_events: List[Dict],
    window_minutes: int = 60,
    audio_timestamp: datetime = None
) -> List[Dict]:
    """
    Find text events that mention or reference an audio clip.
    
    Args:
        audio_hash: Hash of the audio to find mentions of
        text_events: List of text event dicts
        window_minutes: Time window to search (minutes)
        audio_timestamp: Timestamp of the audio (for temporal filtering)
    
    Returns:
        List of dicts containing matched events with confidence scores
    """
    mentions = []
    
    # Audio reference keywords
    audio_keywords = [
        "audio", "sound", "voice", "recording", "listen", "heard",
        "said", "speaking", "music", "clip", "transcript"
    ]
    
    for event in text_events:
        content_lower = event.get('content', '').lower()
        
        # Check for temporal proximity
        if audio_timestamp and 'timestamp' in event:
            try:
                event_time = datetime.fromisoformat(event['timestamp']) if isinstance(event['timestamp'], str) else event['timestamp']
                time_diff = abs((event_time - audio_timestamp).total_seconds() / 60)
                
                if time_diff > window_minutes:
                    continue
                    
                temporal_boost = 1.0 - (time_diff / window_minutes) * 0.3
            except (ValueError, TypeError):
                temporal_boost = 1.0
        else:
            temporal_boost = 1.0
        
        # Check for keywords
        keyword_matches = [kw for kw in audio_keywords if kw in content_lower]
        
        if keyword_matches:
            confidence = 0.6 * temporal_boost + 0.1 * min(len(keyword_matches), 3)
            
            mentions.append({
                'event': event,
                'confidence': min(confidence, 1.0),
                'reason': f"Contains keywords: {', '.join(keyword_matches)}",
                'matched_keywords': keyword_matches
            })
        
        # Direct hash reference
        if audio_hash and audio_hash[:8] in event.get('content', ''):
            mentions.append({
                'event': event,
                'confidence': 0.95,
                'reason': 'Direct hash reference',
                'matched_keywords': []
            })
    
    mentions.sort(key=lambda x: x['confidence'], reverse=True)
    
    logger.info(f"Found {len(mentions)} potential audio mentions in {len(text_events)} text events")
    
    return mentions


def compute_multimodal_relevance(
    query_modality: str,
    result_modality: str,
    similarity_score: float,
    same_session: bool = True
) -> float:
    """
    Compute relevance score for cross-modal retrieval with modality penalties.
    
    Some modality combinations are naturally more related than others.
    For example, text→text is more direct than text→audio.
    
    Args:
        query_modality: Modality of the query
        result_modality: Modality of the result
        similarity_score: Base similarity score (0-1)
        same_session: Whether query and result are from same session
    
    Returns:
        Adjusted relevance score (0-1)
    """
    # Base score
    relevance = similarity_score
    
    # Same modality gets a bonus, no cross-modal penalty
    if query_modality == result_modality:
        relevance *= 1.1
    else:
        # Cross-modal penalties/bonuses (only for different modalities)
        cross_modal_factors = {
            ('text', 'image'): 0.95,  # Text queries for images are common
            ('image', 'text'): 0.95,  # Images can be described by text
            ('text', 'audio'): 0.90,  # Text to audio is less direct
            ('audio', 'text'): 0.95,  # Audio transcripts to text are common
            ('image', 'audio'): 0.85,  # Image to audio is more distant
            ('audio', 'image'): 0.85,
        }
        
        factor = cross_modal_factors.get((query_modality, result_modality), 0.80)
        relevance *= factor
    
    # Session bonus
    if same_session:
        relevance *= 1.05
    
    # Clamp to [0, 1]
    return min(max(relevance, 0.0), 1.0)


def group_by_modality(results: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Group search results by modality.
    
    Args:
        results: List of search result dicts with 'modality' key
    
    Returns:
        Dict mapping modality to list of results
        
    Example:
        >>> results = [{"modality": "text", ...}, {"modality": "image", ...}]
        >>> grouped = group_by_modality(results)
        >>> print(grouped.keys())  # ['text', 'image']
    """
    grouped = {}
    for result in results:
        modality = result.get('modality', 'unknown')
        if modality not in grouped:
            grouped[modality] = []
        grouped[modality].append(result)
    
    return grouped

