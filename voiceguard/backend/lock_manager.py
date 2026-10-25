"""
lock_manager.py — Virtual door lock + access log

The lock starts LOCKED. A verified speaker unlocks it for `unlock_seconds`,
after which it re-locks automatically. Every granted/denied attempt is written
to storage/access_log.jsonl.

To drive real hardware later, act on the state change inside unlock()/lock().
"""
import asyncio
import json
import logging
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("lock_manager")

ACCESS_LOG_PATH = Path(__file__).parent / "storage" / "access_log.jsonl"
RECENT_EVENTS   = 50


class DoorLock:
    def __init__(self):
        self._locked = True
        self._unlocked_by: Optional[dict] = None
        self._relock_at: Optional[float] = None
        self._relock_task: Optional[asyncio.Task] = None
        self._listeners: list[Callable] = []
        self._events: deque = deque(maxlen=RECENT_EVENTS)
        self._load_recent_events()

    # ── State ────────────────────────────────────────────────────────────

    @property
    def is_locked(self) -> bool:
        return self._locked

    def status(self) -> dict:
        return {
            "locked": self._locked,
            "unlocked_by": self._unlocked_by,
            "relock_in": (
                round(max(0.0, self._relock_at - time.monotonic()), 1)
                if self._relock_at else None
            ),
        }

    # ── Actions (call from within the running event loop) ────────────────

    def unlock(self, speaker_id: str, name: str, score: float, seconds: float):
        if not self._locked:
            return
        self._locked = False
        self._unlocked_by = {"speaker_id": speaker_id, "name": name, "score": round(float(score), 4)}
        self._relock_at = time.monotonic() + seconds
        self._relock_task = asyncio.create_task(self._auto_relock(seconds))

        logger.info(f"🔓 UNLOCKED for '{name}' (score {score:.3f}) — re-locks in {seconds:.0f}s")
        self.record("granted", speaker_id=speaker_id, name=name, score=score)
        self._broadcast({"type": "lock", **self.status(), "unlock_seconds": seconds})

    def lock(self, reason: str = "manual"):
        if self._relock_task and not self._relock_task.done() and reason != "auto":
            self._relock_task.cancel()
        was_unlocked = not self._locked
        self._locked = True
        self._unlocked_by = None
        self._relock_at = None
        self._relock_task = None
        if was_unlocked:
            logger.info(f"🔒 Locked ({reason})")
            self._broadcast({"type": "lock", **self.status(), "reason": reason})

    async def _auto_relock(self, seconds: float):
        try:
            await asyncio.sleep(seconds)
            self.lock("auto")
        except asyncio.CancelledError:
            pass

    # ── Access log ───────────────────────────────────────────────────────

    def record(self, result: str, **details):
        """result: 'granted' | 'denied'"""
        event = {"time": datetime.now().isoformat(timespec="seconds"), "result": result}
        for k, v in details.items():
            event[k] = round(float(v), 4) if isinstance(v, float) else v
        self._events.appendleft(event)
        try:
            ACCESS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(ACCESS_LOG_PATH, "a") as f:
                f.write(json.dumps(event) + "\n")
        except OSError as e:
            logger.error(f"Could not write access log: {e}")
        if result == "denied":
            logger.info(f"⛔ Access denied: {details.get('reason')}")
        self._broadcast({"type": "access", "event": event})

    def recent_events(self) -> list[dict]:
        return list(self._events)

    def _load_recent_events(self):
        if not ACCESS_LOG_PATH.exists():
            return
        try:
            lines = ACCESS_LOG_PATH.read_text().splitlines()[-RECENT_EVENTS:]
            for line in lines:
                self._events.appendleft(json.loads(line))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Could not read access log: {e}")

    # ── Listeners (WebSocket connections) ────────────────────────────────

    def add_listener(self, callback: Callable):
        self._listeners.append(callback)

    def remove_listener(self, callback: Callable):
        if callback in self._listeners:
            self._listeners.remove(callback)

    def _broadcast(self, message: dict):
        for cb in list(self._listeners):
            asyncio.create_task(self._safe_send(cb, message))

    @staticmethod
    async def _safe_send(cb: Callable, message: dict):
        try:
            await cb(message)
        except Exception as e:
            logger.debug(f"Listener error: {e}")


# Global singleton
door = DoorLock()
