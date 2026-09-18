from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum


class RequestStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"


@dataclass(frozen=True)
class PrimeRequest:
    action: str
    payload: dict[str, Any] = field(default_factory=dict)
    context: Optional[dict[str, Any]] = None
    metadata: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class PrimeResult:
    status: RequestStatus
    output: Optional[Any] = None
    error: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None

    @classmethod
    def success(cls, output: Any = None, metadata: Optional[dict[str, Any]] = None) -> "PrimeResult":
        return cls(status=RequestStatus.SUCCESS, output=output, metadata=metadata)

    @classmethod
    def failure(cls, error: str, metadata: Optional[dict[str, Any]] = None) -> "PrimeResult":
        return cls(status=RequestStatus.FAILURE, error=error, metadata=metadata)