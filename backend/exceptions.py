"""
Йерархия от специфични грешки на /backend.

    BackendError
    ├── InvalidTickerError       (ValueError)   – тикер извън фиксираните 20
    ├── InvalidDateRangeError    (ValueError)   – период извън 2021-01-01 – 2025-12-31
    ├── DataValidationError      (ValueError)   – невалиден OHLCV ред (> 3 липсващи дни,
    │                                             нарушени OHLC инварианти и др.)
    └── ExternalDataFetchError   (RuntimeError) – неуспешно теглене от Yahoo Finance

`contracts/CONTRACTS.md` §1 изисква при > 3 последователни липсващи търговски дни да се
хвърля `ValueError` – затова валидационните грешки наследяват и `ValueError`; код,
който хваща `ValueError`, продължава да работи.

`status_code` е подсказка за бъдещите FastAPI exception handlers (Етап 3).
"""

from __future__ import annotations

from typing import Any


class BackendError(Exception):
    """Базова грешка на /backend."""

    status_code: int = 500

    def __init__(self, message: str, *, ticker: str | None = None, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.ticker = ticker
        self.details: dict[str, Any] = details

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"error": type(self).__name__, "message": self.message}
        if self.ticker is not None:
            payload["ticker"] = self.ticker
        if self.details:
            payload["details"] = self.details
        return payload


class InvalidTickerError(BackendError, ValueError):
    """Тикерът не е сред 20-те разрешени актива."""

    status_code = 422


class InvalidDateRangeError(BackendError, ValueError):
    """Периодът е обърнат или излиза извън 2021-01-01 – 2025-12-31."""

    status_code = 422


class DataValidationError(BackendError, ValueError):
    """Времевият ред е невалиден и симулацията трябва да спре."""

    status_code = 422


class ExternalDataFetchError(BackendError, RuntimeError):
    """Yahoo Finance е недостъпен или върна празни/неразпознаваеми данни."""

    status_code = 502


__all__ = [
    "BackendError",
    "DataValidationError",
    "ExternalDataFetchError",
    "InvalidDateRangeError",
    "InvalidTickerError",
]
