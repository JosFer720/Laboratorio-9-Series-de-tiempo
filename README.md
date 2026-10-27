# Laboratorio de series de tiempo

Pipeline reproducible para pronosticar 24 horas de consumo eléctrico por cliente con el conjunto ElectricityLoadDiagrams20112014.

## Entorno

Use Python 3.11: varias dependencias de forecasting todavía no ofrecen wheels para Python 3.13.

```bash
python -m venv .venv
.venv/Scripts/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Verificación

```bash
python -m pytest -q
python scripts/build_notebook.py
python scripts/build_report.py
```

## Flujo reproducible

1. La carga valida el índice, agrega cuatro cuartos de hora a kWh y filtra clientes completos desde 2012.
2. Los splits, el normalizador y las covariables se definen una sola vez en `src/forecasting_lab`.
3. `PanelWindowDataset` genera ventanas `(L, atributos)` y objetivos `(H,)` sin materializarlas por adelantado.
4. `backtest` aplica los 92 orígenes diarios y guarda pronósticos con `unique_id, ds, modelo, y_hat`.
5. Los adaptadores opcionales aíslan StatsForecast, LightGBM, NeuralForecast y Chronos.

El notebook principal es `laboratorio_series_tiempo.ipynb`. Los resultados generados se guardan en `artifacts/` y el reporte en `report/reporte_laboratorio.pdf`.
