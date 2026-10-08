"""Phase 4 clarify-once session logic."""

from __future__ import annotations
import json
from pathlib import Path
import re

class ClarifySession:
    """Simple interactive state machine for ambiguous queries with persistent memory."""

    def __init__(self) -> None:
        self.is_running = True
        self.kb_path = Path("data") / "knowledge_base.json"
        self.kb = self._load_kb()

    def _load_kb(self):
        if self.kb_path.exists():
            try:
                with open(self.kb_path, "r") as f:
                    return json.load(f)
            except:
                pass
        return {}

    def _save_kb(self):
        self.kb_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.kb_path, "w") as f:
            json.dump(self.kb, f, indent=2)

    def resolve_query(self, query: str) -> str:
        resolved = query
        for entity, description in self.kb.items():
            resolved = re.sub(r'\b' + re.escape(entity) + r'\b', description, resolved, flags=re.IGNORECASE)
        return resolved

    def clarify_once(self, query: str) -> str | None:
        if not self.is_running:
            return None
            
        # Hardcode some entities that might need clarification if they are not in the KB
        potential_entities = ["main gate", "lobby", "rear cam", "parking lot"]
        for entity in potential_entities:
            if re.search(r'\b' + re.escape(entity) + r'\b', query, re.IGNORECASE):
                if entity.lower() not in self.kb:
                    return f"Which one is the {entity}? (Please provide a description, e.g., 'camera 1 region')"

        query_lower = query.lower()
        if " and " in query_lower or " or " in query_lower or "," in query_lower:
            return "Do you want me to search for one object or multiple objects?"
        return None
        
    def learn(self, query: str, answer: str) -> None:
        """Learn and persist a spatial referent mapping permanently to data/knowledge_base.json."""
        clean_query = query.strip()
        clean_ans = answer.strip()
        if not clean_query or not clean_ans:
            return

        # Check if query is directly an entity name
        clean_entity = clean_query.lower()
        potential_entities = ["main gate", "lobby", "rear cam", "parking lot", "rear exit", "back gate", "front gate", "entrance", "exit", "gate"]

        matched = False
        for entity in potential_entities:
            if re.search(r'\b' + re.escape(entity) + r'\b', clean_query, re.IGNORECASE):
                self.kb[entity.lower()] = clean_ans
                matched = True

        if not matched:
            # Save whatever entity was passed
            self.kb[clean_entity] = clean_ans

        self._save_kb()

    def learn_mapping(self, entity: str, target: str) -> None:
        """Directly register a spatial referent to a camera or region."""
        ent = entity.strip().lower()
        tgt = target.strip()
        if ent and tgt:
            self.kb[ent] = tgt
            self._save_kb()

    def delete_mapping(self, entity: str) -> bool:
        """Remove a learned entity from persistent memory."""
        ent = entity.strip().lower()
        if ent in self.kb:
            del self.kb[ent]
            self._save_kb()
            return True
        return False

    def get_mappings(self) -> dict[str, str]:
        """Return all persisted spatial referents."""
        return dict(self.kb)

    def kill(self) -> None:
        self.is_running = False

    def restart(self) -> None:
        self.is_running = True
        self.kb = self._load_kb()


