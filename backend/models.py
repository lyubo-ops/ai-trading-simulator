"""
Релационна схема (PostgreSQL) чрез SQLAlchemy 2.0 ORM, адаптирана 1:1 към
официалния договор `contracts/mock_backtest_response.json`.

Таблици:
    market_data       – почистени дневни OHLCV редове; уникален (ticker, date)
    simulation_runs   – request_params + ml_metrics + portfolio_metrics (ai_strategy,
                        buy_and_hold) + equity_curve
    audit_logs        – audit_logs от договора (FK -> simulation_runs, ON DELETE CASCADE)

Съответствие с договора:
    request_params.*                         -> SimulationRun.<същото име>
    ml_metrics.*                             -> SimulationRun.<същото име>
    portfolio_metrics.ai_strategy.<x>        -> SimulationRun.ai_<x>
    portfolio_metrics.buy_and_hold.<x>       -> SimulationRun.bnh_<x>
    equity_curve                             -> SimulationRun.equity_curve (JSON, както е)
    audit_logs[i].<x>                        -> AuditLog.<същото име>

Единици: `max_position_pct` и `max_drawdown_limit` са дроби (0.25, 0.15);
метриките с наставка `_pct` са в проценти (42.5 == 42.5 %).

Модел на изпълнение (официалното задание):
    * Сигналите BUY/SELL се генерират след Close(t) (signal_date = t) и се
      изпълняват във ФАЗА 1 на следващия ден по Open(t+1) (execution_date = t+1).
    * FORCE_SELL: пробив на Max Drawdown при Close(t) -> на опашка -> изпълнение по Open(t+1).
    * FORCE_SELL_AT_OPEN: нощен гап при Open(t) -> незабавно изпълнение по Open(t)
      (signal_date = execution_date = t).
    * REJECTED_BY_RISK (само за BUY): във фаза CLOSE (блокиран нов сигнал) или
      във фаза OPEN (отменен чакащ BUY от t-1 заради гап, задействал прекъсвача).
    * Всяко реално изпълнение (status = EXECUTED) е във фаза OPEN.
"""

from __future__ import annotations

import enum
import hashlib
import json
import uuid
from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    CheckConstraint,
    Date,
    DateTime,
    Double,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    Uuid,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base

JSONType = JSON().with_variant(JSONB(), "postgresql")
# BIGINT в PostgreSQL; INTEGER в SQLite, за да работи AUTOINCREMENT в тестовете
BigIntPK = BigInteger().with_variant(Integer(), "sqlite")


# ==============================================================================
# Енумерации (стойностите съвпадат с низовете в договора)
# ==============================================================================


class AuditAction(str, enum.Enum):
    """`audit_logs[].action`"""

    BUY = "BUY"
    SELL = "SELL"
    FORCE_SELL = "FORCE_SELL"                  # пробив при Close(t) -> изпълнение по Open(t+1)
    FORCE_SELL_AT_OPEN = "FORCE_SELL_AT_OPEN"  # пробив от нощен гап -> изпълнение по Open(t)


class AuditStatus(str, enum.Enum):
    """`audit_logs[].status`"""

    EXECUTED = "EXECUTED"
    REJECTED_BY_RISK = "REJECTED_BY_RISK"


class Phase(str, enum.Enum):
    """Фаза на двуфазния двигател, в която е възникнал записът (поле само в БД)."""

    OPEN = "OPEN"    # Фаза 1 – Open(t)
    CLOSE = "CLOSE"  # Фаза 2 – Close(t)


class SimulationStatus(str, enum.Enum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


def _pg_enum(enum_cls: type[enum.Enum], name: str) -> Enum:
    """Native PostgreSQL ENUM със стойностите (а не имената) на Python enum-а."""
    return Enum(
        enum_cls,
        name=name,
        values_callable=lambda e: [m.value for m in e],
        validate_strings=True,
    )


def _iso(d: date | None) -> str | None:
    return d.isoformat() if d is not None else None


def _date(s: str | None) -> date | None:
    return date.fromisoformat(s) if s is not None else None


# ==============================================================================
# MarketData
# ==============================================================================


class MarketData(Base):
    """Почистен дневен OHLCV ред (contracts/CONTRACTS.md §2)."""

    __tablename__ = "market_data"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)

    open: Mapped[float] = mapped_column(Double, nullable=False)
    high: Mapped[float] = mapped_column(Double, nullable=False)
    low: Mapped[float] = mapped_column(Double, nullable=False)
    close: Mapped[float] = mapped_column(Double, nullable=False)
    volume: Mapped[float] = mapped_column(Double, nullable=False)  # float64 по договора

    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_market_data_ticker_date", "ticker", "date", unique=True),
        CheckConstraint("open > 0 AND high > 0 AND low > 0 AND close > 0", name="positive_prices"),
        CheckConstraint("high >= low", name="high_gte_low"),
        CheckConstraint("open >= low AND open <= high", name="open_within_range"),
        CheckConstraint("close >= low AND close <= high", name="close_within_range"),
        CheckConstraint("volume >= 0", name="non_negative_volume"),
    )

    def __repr__(self) -> str:
        return f"<MarketData {self.ticker} {self.date} C={self.close}>"


# ==============================================================================
# SimulationRun
# ==============================================================================


class SimulationRun(Base):
    """Една пълна симулация (ML + backtest) за един актив."""

    __tablename__ = "simulation_runs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    status: Mapped[SimulationStatus] = mapped_column(
        _pg_enum(SimulationStatus, "simulation_status"),
        nullable=False,
        default=SimulationStatus.COMPLETED,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # --- request_params ---------------------------------------------------------
    ticker: Mapped[str] = mapped_column(String(16), nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)
    initial_capital: Mapped[float] = mapped_column(Double, nullable=False)
    t_buy: Mapped[float] = mapped_column(Double, nullable=False)
    t_sell: Mapped[float] = mapped_column(Double, nullable=False)
    max_position_pct: Mapped[float] = mapped_column(Double, nullable=False)
    max_drawdown_limit: Mapped[float] = mapped_column(Double, nullable=False)
    # Вътрешни параметри за възпроизводимост (не са част от request_params)
    commission: Mapped[float] = mapped_column(Double, nullable=False, default=0.001)
    random_state: Mapped[int] = mapped_column(Integer, nullable=False, default=42)
    params_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    # --- ml_metrics -------------------------------------------------------------
    accuracy: Mapped[float | None] = mapped_column(Double)
    precision: Mapped[float | None] = mapped_column(Double)
    recall: Mapped[float | None] = mapped_column(Double)
    f1_score: Mapped[float | None] = mapped_column(Double)

    # --- portfolio_metrics.ai_strategy ------------------------------------------
    ai_final_equity: Mapped[float | None] = mapped_column(Double)
    ai_net_return_pct: Mapped[float | None] = mapped_column(Double)
    ai_max_drawdown_pct: Mapped[float | None] = mapped_column(Double)
    ai_sharpe_ratio: Mapped[float | None] = mapped_column(Double)
    ai_win_rate_pct: Mapped[float | None] = mapped_column(Double)
    ai_total_trades: Mapped[int | None] = mapped_column(Integer)
    ai_circuit_breaker_triggered: Mapped[bool | None] = mapped_column(default=None)

    # --- portfolio_metrics.buy_and_hold -----------------------------------------
    bnh_final_equity: Mapped[float | None] = mapped_column(Double)
    bnh_net_return_pct: Mapped[float | None] = mapped_column(Double)
    bnh_max_drawdown_pct: Mapped[float | None] = mapped_column(Double)
    bnh_sharpe_ratio: Mapped[float | None] = mapped_column(Double)

    # --- equity_curve (списък от {date, close_price, ai_equity_close,
    #     bnh_equity_close, drawdown_close_pct} – съхранява се както е) -----------
    equity_curve: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONType, nullable=False, default=list
    )

    # --- Технически -------------------------------------------------------------
    execution_time_ms: Mapped[int | None] = mapped_column(Integer)
    error_message: Mapped[str | None] = mapped_column(Text)

    audit_logs: Mapped[list["AuditLog"]] = relationship(
        back_populates="simulation_run",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="AuditLog.sequence",
        lazy="selectin",
    )

    __table_args__ = (
        Index("ix_simulation_runs_ticker_created_at", "ticker", "created_at"),
        Index("ix_simulation_runs_params_hash", "params_hash"),
        CheckConstraint("start_date < end_date", name="valid_period"),
        CheckConstraint("initial_capital > 0", name="positive_capital"),
        CheckConstraint("t_sell >= 0 AND t_buy <= 1 AND t_sell < t_buy", name="valid_thresholds"),
        CheckConstraint("max_position_pct > 0 AND max_position_pct <= 1", name="valid_position_pct"),
        CheckConstraint(
            "max_drawdown_limit > 0 AND max_drawdown_limit < 1", name="valid_mdd_limit"
        ),
        CheckConstraint("commission >= 0 AND commission < 1", name="valid_commission"),
        CheckConstraint("random_state >= 0", name="valid_random_state"),
        CheckConstraint(
            "(accuracy IS NULL OR (accuracy >= 0 AND accuracy <= 1)) AND "
            "(precision IS NULL OR (precision >= 0 AND precision <= 1)) AND "
            "(recall IS NULL OR (recall >= 0 AND recall <= 1)) AND "
            "(f1_score IS NULL OR (f1_score >= 0 AND f1_score <= 1))",
            name="valid_ml_metrics",
        ),
        CheckConstraint(
            "(ai_max_drawdown_pct IS NULL OR (ai_max_drawdown_pct >= 0 AND ai_max_drawdown_pct <= 100)) AND "
            "(bnh_max_drawdown_pct IS NULL OR (bnh_max_drawdown_pct >= 0 AND bnh_max_drawdown_pct <= 100)) AND "
            "(ai_win_rate_pct IS NULL OR (ai_win_rate_pct >= 0 AND ai_win_rate_pct <= 100)) AND "
            "(ai_net_return_pct IS NULL OR ai_net_return_pct >= -100) AND "
            "(bnh_net_return_pct IS NULL OR bnh_net_return_pct >= -100)",
            name="valid_pct_metrics",
        ),
        CheckConstraint("ai_total_trades IS NULL OR ai_total_trades >= 0", name="valid_total_trades"),
    )

    # --------------------------------------------------------------------------
    # Преобразуване от/към формата на contracts/mock_backtest_response.json
    # --------------------------------------------------------------------------

    @staticmethod
    def compute_params_hash(request_params: dict[str, Any], *, commission: float,
                            random_state: int) -> str:
        """SHA-256 на каноничния JSON на входа – еднакъв вход => еднакъв хеш."""
        canonical = json.dumps(
            {**request_params, "commission": commission, "random_state": random_state},
            sort_keys=True, separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def from_contract(
        cls,
        payload: dict[str, Any],
        *,
        commission: float = 0.001,
        random_state: int = 42,
        execution_time_ms: int | None = None,
    ) -> "SimulationRun":
        """Създава SimulationRun + AuditLog записи от отговора на /backtester."""
        rp = payload["request_params"]
        ml = payload.get("ml_metrics") or {}
        ai = (payload.get("portfolio_metrics") or {}).get("ai_strategy") or {}
        bnh = (payload.get("portfolio_metrics") or {}).get("buy_and_hold") or {}

        run = cls(
            ticker=rp["ticker"],
            start_date=date.fromisoformat(rp["start_date"]),
            end_date=date.fromisoformat(rp["end_date"]),
            initial_capital=rp["initial_capital"],
            t_buy=rp["t_buy"],
            t_sell=rp["t_sell"],
            max_position_pct=rp["max_position_pct"],
            max_drawdown_limit=rp["max_drawdown_limit"],
            commission=commission,
            random_state=random_state,
            params_hash=cls.compute_params_hash(rp, commission=commission,
                                                random_state=random_state),
            accuracy=ml.get("accuracy"),
            precision=ml.get("precision"),
            recall=ml.get("recall"),
            f1_score=ml.get("f1_score"),
            ai_final_equity=ai.get("final_equity"),
            ai_net_return_pct=ai.get("net_return_pct"),
            ai_max_drawdown_pct=ai.get("max_drawdown_pct"),
            ai_sharpe_ratio=ai.get("sharpe_ratio"),
            ai_win_rate_pct=ai.get("win_rate_pct"),
            ai_total_trades=ai.get("total_trades"),
            ai_circuit_breaker_triggered=ai.get("circuit_breaker_triggered"),
            bnh_final_equity=bnh.get("final_equity"),
            bnh_net_return_pct=bnh.get("net_return_pct"),
            bnh_max_drawdown_pct=bnh.get("max_drawdown_pct"),
            bnh_sharpe_ratio=bnh.get("sharpe_ratio"),
            equity_curve=list(payload.get("equity_curve") or []),
            execution_time_ms=execution_time_ms,
        )
        run.audit_logs = [
            AuditLog.from_contract(entry, sequence=i)
            for i, entry in enumerate(payload.get("audit_logs") or [])
        ]
        return run

    def to_contract(self) -> dict[str, Any]:
        """Сериализира обратно в точния формат на mock_backtest_response.json."""
        return {
            "request_params": {
                "ticker": self.ticker,
                "start_date": self.start_date.isoformat(),
                "end_date": self.end_date.isoformat(),
                "initial_capital": self.initial_capital,
                "t_buy": self.t_buy,
                "t_sell": self.t_sell,
                "max_position_pct": self.max_position_pct,
                "max_drawdown_limit": self.max_drawdown_limit,
            },
            "ml_metrics": {
                "accuracy": self.accuracy,
                "precision": self.precision,
                "recall": self.recall,
                "f1_score": self.f1_score,
            },
            "portfolio_metrics": {
                "ai_strategy": {
                    "final_equity": self.ai_final_equity,
                    "net_return_pct": self.ai_net_return_pct,
                    "max_drawdown_pct": self.ai_max_drawdown_pct,
                    "sharpe_ratio": self.ai_sharpe_ratio,
                    "win_rate_pct": self.ai_win_rate_pct,
                    "total_trades": self.ai_total_trades,
                    "circuit_breaker_triggered": self.ai_circuit_breaker_triggered,
                },
                "buy_and_hold": {
                    "final_equity": self.bnh_final_equity,
                    "net_return_pct": self.bnh_net_return_pct,
                    "max_drawdown_pct": self.bnh_max_drawdown_pct,
                    "sharpe_ratio": self.bnh_sharpe_ratio,
                },
            },
            "equity_curve": list(self.equity_curve),
            "audit_logs": [log.to_contract() for log in self.audit_logs],
        }

    def __repr__(self) -> str:
        return f"<SimulationRun {self.id} {self.ticker} {self.start_date}..{self.end_date}>"


# ==============================================================================
# AuditLog
# ==============================================================================


class AuditLog(Base):
    """
    Един запис от `audit_logs` на договора + вътрешните полета `sequence` и `phase`.

    Допустими комбинации (налагат се от CHECK ограничения):

    | action             | status           | phase | execution_date      | price / units | prob_up  |
    |--------------------|------------------|-------|---------------------|---------------|----------|
    | BUY / SELL         | EXECUTED         | OPEN  | > signal_date (t+1) | > 0 / > 0     | NOT NULL |
    | FORCE_SELL         | EXECUTED         | OPEN  | > signal_date (t+1) | > 0 / > 0     | NULL     |
    | FORCE_SELL_AT_OPEN | EXECUTED         | OPEN  | = signal_date (t)   | > 0 / > 0     | NULL     |
    | BUY                | REJECTED_BY_RISK | CLOSE | NULL или > signal   | NULL / 0      | NOT NULL |
    | BUY                | REJECTED_BY_RISK | OPEN  | > signal_date (t+1) | NULL / 0      | NOT NULL |
    """

    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    simulation_run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("simulation_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    # Поредност в рамките на симулацията (= индексът в масива audit_logs)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)

    # --- Полета от договора ------------------------------------------------------
    signal_date: Mapped[date] = mapped_column(Date, nullable=False)
    execution_date: Mapped[date | None] = mapped_column(Date)
    prob_up: Mapped[float | None] = mapped_column(Double)
    action: Mapped[AuditAction] = mapped_column(
        _pg_enum(AuditAction, "audit_action"), nullable=False
    )
    status: Mapped[AuditStatus] = mapped_column(
        _pg_enum(AuditStatus, "audit_status"), nullable=False
    )
    execution_price: Mapped[float | None] = mapped_column(Double)
    units: Mapped[float] = mapped_column(Double, nullable=False, default=0.0)
    fee_paid: Mapped[float] = mapped_column(Double, nullable=False, default=0.0)
    reason: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # --- Вътрешно поле: фаза на двигателя -----------------------------------------
    phase: Mapped[Phase] = mapped_column(_pg_enum(Phase, "simulation_phase"), nullable=False)

    simulation_run: Mapped[SimulationRun] = relationship(back_populates="audit_logs")

    __table_args__ = (
        Index("ix_audit_logs_run_sequence", "simulation_run_id", "sequence", unique=True),
        Index("ix_audit_logs_run_signal_date", "simulation_run_id", "signal_date"),
        Index("ix_audit_logs_status", "status"),
        CheckConstraint("sequence >= 0", name="non_negative_sequence"),
        CheckConstraint("units >= 0", name="non_negative_units"),
        CheckConstraint("fee_paid >= 0", name="non_negative_fee"),
        CheckConstraint("prob_up IS NULL OR (prob_up >= 0 AND prob_up <= 1)", name="valid_prob_up"),
        # Реално изпълнение има само във Фаза 1 (Open) – с цена, количество и дата;
        # отхвърлянето засяга само BUY и е без изпълнение.
        CheckConstraint(
            "(status = 'EXECUTED' AND phase = 'OPEN' AND execution_date IS NOT NULL "
            "  AND execution_price IS NOT NULL AND execution_price > 0 AND units > 0) OR "
            "(status = 'REJECTED_BY_RISK' AND action = 'BUY' "
            "  AND execution_price IS NULL AND units = 0 AND fee_paid = 0)",
            name="execution_lifecycle",
        ),
        # Времева логика Open(t+1):
        #   BUY/SELL/FORCE_SELL: сигнал след Close(t), изпълнение по Open(t+1) -> по-късна дата;
        #   FORCE_SELL_AT_OPEN: гап при Open(t), изпълнение в същия ден;
        #   REJECTED във фаза OPEN: отменен чакащ BUY -> execution_date = t+1;
        #   REJECTED във фаза CLOSE: блокиран при генериране -> планирана дата t+1 или NULL.
        CheckConstraint(
            "(status = 'EXECUTED' AND action = 'FORCE_SELL_AT_OPEN' "
            "  AND execution_date = signal_date) OR "
            "(status = 'EXECUTED' AND action IN ('BUY', 'SELL', 'FORCE_SELL') "
            "  AND execution_date > signal_date) OR "
            "(status = 'REJECTED_BY_RISK' AND phase = 'OPEN' "
            "  AND execution_date IS NOT NULL AND execution_date > signal_date) OR "
            "(status = 'REJECTED_BY_RISK' AND phase = 'CLOSE' "
            "  AND (execution_date IS NULL OR execution_date > signal_date))",
            name="open_t_plus_1_timing",
        ),
        # Вероятност има само при ML сигнали; принудителните продажби идват от Risk модула
        CheckConstraint(
            "(action IN ('FORCE_SELL', 'FORCE_SELL_AT_OPEN') AND prob_up IS NULL) OR "
            "(action IN ('BUY', 'SELL') AND prob_up IS NOT NULL)",
            name="prob_up_presence",
        ),
    )

    # --------------------------------------------------------------------------

    @staticmethod
    def infer_phase(entry: dict[str, Any]) -> Phase:
        """
        Договорът няма поле `phase`; извежда се така:
          * EXECUTED -> OPEN (всички изпълнения са по Open);
          * REJECTED_BY_RISK -> стойността на незадължителния ключ `phase`, ако го има,
            иначе CLOSE (блокиран сигнал след Close(t) – случаят в mock файла).
        """
        if entry["status"] == AuditStatus.EXECUTED.value:
            return Phase.OPEN
        return Phase(entry.get("phase", Phase.CLOSE.value))

    @classmethod
    def from_contract(cls, entry: dict[str, Any], *, sequence: int) -> "AuditLog":
        return cls(
            sequence=sequence,
            signal_date=date.fromisoformat(entry["signal_date"]),
            execution_date=_date(entry.get("execution_date")),
            prob_up=entry.get("prob_up"),
            action=AuditAction(entry["action"]),
            status=AuditStatus(entry["status"]),
            execution_price=entry.get("execution_price"),
            units=entry.get("units", 0.0),
            fee_paid=entry.get("fee_paid", 0.0),
            reason=entry.get("reason", ""),
            phase=cls.infer_phase(entry),
        )

    def to_contract(self) -> dict[str, Any]:
        return {
            "signal_date": self.signal_date.isoformat(),
            "execution_date": _iso(self.execution_date),
            "prob_up": self.prob_up,
            "action": self.action.value,
            "status": self.status.value,
            "execution_price": self.execution_price,
            "units": self.units,
            "fee_paid": self.fee_paid,
            "reason": self.reason,
        }

    def __repr__(self) -> str:
        return (
            f"<AuditLog run={self.simulation_run_id} #{self.sequence} {self.signal_date}->"
            f"{self.execution_date} {self.action.value} {self.status.value} {self.phase.value}>"
        )


__all__ = [
    "AuditAction",
    "AuditLog",
    "AuditStatus",
    "MarketData",
    "Phase",
    "SimulationRun",
    "SimulationStatus",
]
