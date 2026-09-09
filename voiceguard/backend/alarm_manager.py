"""
alarm_manager.py — Alarm state management
"""
import asyncio
import logging
from datetime import datetime
from typing import Callable, Optional

logger = logging.getLogger("alarm_manager")


class AlarmManager:
    COOLDOWN_SECONDS = 5  # seconds before alarm can retrigger after being stopped

    def __init__(self):
        self._active = False
        self._triggered_at: Optional[str] = None
        self._triggered_by: Optional[str] = None
        self._listeners: list[Callable] = []
        self._cooldown_until: float = 0.0  # epoch seconds

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def can_trigger(self) -> bool:
        """Returns True if alarm is not active and cooldown has elapsed."""
        import time
        return not self._active and time.time() >= self._cooldown_until

    def trigger(self, speaker_id: str, dl_score: float, dtw_score: float):
        if not self.can_trigger:
            return  # active or in cooldown
        self._active = True
        self._triggered_at = datetime.now().isoformat()
        self._triggered_by = speaker_id
        logger.info(f"🚨 ALARM triggered by '{speaker_id}' — DL: {dl_score:.3f}, DTW: {dtw_score:.3f}")
        asyncio.create_task(self._notify_listeners({
            "type": "alarm",
            "active": True,
            "speaker": speaker_id,
            "dl_score": dl_score,
            "dtw_score": dtw_score,
            "triggered_at": self._triggered_at
        }))

    def stop(self):
        import time
        self._active = False
        self._cooldown_until = time.time() + self.COOLDOWN_SECONDS
        logger.info(f"🔕 Alarm stopped. Cooldown for {self.COOLDOWN_SECONDS}s.")
        asyncio.create_task(self._notify_listeners({
            "type": "alarm",
            "active": False
        }))

    def add_listener(self, callback: Callable):
        self._listeners.append(callback)

    def remove_listener(self, callback: Callable):
        self._listeners.discard(callback) if hasattr(self._listeners, 'discard') else None
        if callback in self._listeners:
            self._listeners.remove(callback)

    async def _notify_listeners(self, message: dict):
        for cb in list(self._listeners):
            try:
                await cb(message)
            except Exception as e:
                logger.error(f"Listener error: {e}")

    def status(self) -> dict:
        return {
            "active": self._active,
            "triggered_at": self._triggered_at,
            "triggered_by": self._triggered_by
        }


# Global singleton
alarm = AlarmManager()
