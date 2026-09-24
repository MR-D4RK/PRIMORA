from enum import Enum


class VerseLifecycleState(Enum):
    STARTING = "starting"
    AWAKENING = "awakening"
    AWAKE = "awake"
    ASLEEP = "asleep"
    CLOSING = "closing"
    CLOSED = "closed"
    DEGRADED = "degraded"


class PresentationMode(Enum):
    IMMERSIVE = "immersive"
    PORTAL = "portal"
