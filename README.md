# AI-Powered Investment Platform for Simulated Algorithmic Trading

Университетски проект за анализ и симулирана алгоритмична търговия върху исторически данни (01.01.2021 – 31.12.2025) за 20 фиксирани актива (10 крипто и 10 акции от S&P 500).

## Структура на проекта и разпределение
* `/backend` – FastAPI сървър, PostgreSQL база данни и извличане на данни от Yahoo Finance (**Искамил**)
* `/ai_engine` – Технически индикатори, Random Forest класификатор и Walk-Forward оценка (**Любомир**)
* `/backtester` – Двуфазен симулационен двигател (Open/Close), Risk Management модул и документация (**Дарио**)
* `/frontend` – React + TypeScript потребителски интерфейс и графики (**Калоян**)
* `/contracts` – Договори за форматите на данните и примерни (mock) файлове за независима работа
* `/tests` – Автоматизирани Pytest сценарии

## Фиксиран списък с 20 тикера (Yahoo Finance)
* **Крипто (10):** `BTC-USD`, `ETH-USD`, `BNB-USD`, `XRP-USD`, `LTC-USD`, `ADA-USD`, `DOT-USD`, `LINK-USD`, `BCH-USD`, `XLM-USD`
* **Акции S&P 500 (10):** `AAPL`, `MSFT`, `AMZN`, `GOOGL`, `META`, `TSLA`, `BRK-B`, `JPM`, `JNJ`, `V`

## Правила за работа с Git
1. Никой не качва директно в клона `main`.
2. Всеки работи в собствен клон: `git checkout -b feature/име-на-модула`.
3. Преди промяна на входни/изходни данни между модулите, актуализираме файла `contracts/CONTRACTS.md`.