"""Genera el notebook de la sección 5 (entrenamiento) a partir de los artefactos."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ANALYSIS_PATH = Path(__file__).with_name("analisis_entrenamiento.md")


def _analysis(section: str) -> str:
    """Texto de análisis por sección, escrito después de revisar los resultados."""

    if not ANALYSIS_PATH.exists():
        return ""
    blocks = ANALYSIS_PATH.read_text(encoding="utf-8").split("<!-- SECCION: ")
    for block in blocks[1:]:
        name, _, body = block.partition(" -->")
        if name.strip() == section:
            return body.strip()
    return ""


def build_notebook(output: str | Path = "entrenamiento.ipynb") -> Path:
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [
        md(
            """# Sección 5: construcción y entrenamiento de las redes

Este notebook entrena y evalúa las redes del laboratorio sobre el mismo panel de 100 clientes, la misma semilla y la misma función `backtest`. Las corridas largas se ejecutan por etapas con `scripts/run_entrenamiento.py` (cada etapa guarda CSV en `artifacts/`) y aquí se leen esos artefactos; con `EJECUTAR_CORRIDAS = True` las celdas relanzan las etapas.

| Etapa | Comando |
|---|---|
| Iteraciones 5.2 (solo validación) | `python scripts/run_entrenamiento.py iteraciones` |
| Prueba de las redes propias | `python scripts/run_entrenamiento.py prueba-propias` |
| NeuralForecast (N-HiTS, TCN, PatchTST, iTransformer, DeepAR) | `python scripts/run_entrenamiento.py nixtla` |
| PatchTST con 3 y 12 meses | `python scripts/run_entrenamiento.py patchtst-historia` |
| Sensibilidad a la historia | `python scripts/run_entrenamiento.py historia` |
| Local contra global (5.3) | `python scripts/run_entrenamiento.py local-global` |
| Tres semillas | `python scripts/run_entrenamiento.py semillas` |
| Reentrenamiento mensual | `python scripts/run_entrenamiento.py mensual` |

**Decisiones comunes.**
- Ventanas de entrenamiento con stride de 1 hora cuyo objetivo completo cae en train (hasta 2014-06-30 23:00), muestreadas directamente en GPU.
- Pérdida L1 en escala normalizada; Adam, recorte de gradiente a 1 y early stopping por MASE de validación (paciencia 4, máximo 25 epochs).
- La validación usa los 92 orígenes diarios de julio a septiembre, con el mismo protocolo que la prueba.
- La prueba se corre una sola vez por modelo con `backtest()` y los pesos elegidos en validación.
- Los análisis repetidos (semillas, local/global, mensual) usan `rolling_forecasts`, el evaluador vectorizado que `tests/test_training.py` verifica contra `backtest`."""
        ),
        code(
            """from pathlib import Path
import subprocess, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from forecasting_lab.config import ExperimentConfig
from forecasting_lab.results import plot_history_sensitivity
from forecasting_lab.training import (PanelTensors, daily_origins, predict_positions,
                                      rolling_forecasts, forecast_errors)

sys.path.insert(0, "scripts")
from run_entrenamiento import base_configs, load_panel, load_run, run_id_of

CONFIG = ExperimentConfig()
EJECUTAR_CORRIDAS = False
ARTEFACTOS = Path("artifacts")
pd.set_option("display.max_colwidth", 90)
pd.set_option("display.width", 200)

if EJECUTAR_CORRIDAS:
    for etapa in ["iteraciones", "prueba-propias", "nixtla", "patchtst-historia", "historia",
                  "local-global", "semillas", "mensual"]:
        subprocess.run([sys.executable, "scripts/run_entrenamiento.py", etapa], check=True)

panel = load_panel(ARTEFACTOS)
iteraciones = pd.read_csv(ARTEFACTOS / "iteraciones_entrenamiento.csv")
resumen = pd.read_csv(ARTEFACTOS / "iteraciones_resumen.csv")
resultados = pd.read_csv(ARTEFACTOS / "resultados_redes.csv")
print(panel.shape, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no disponible")"""
        ),
        md(
            """## 5. Construcción y entrenamiento

Las redes propias usan las clases de `models.py`:
- **MLP multi-salida.** Recibe el objetivo y las 7 covariables de calendario de las `L` horas.
- **DLinear y NLinear.** Son univariados.
- **LSTM multi-salida.** Una capa densa devuelve los 24 pasos.
- **LSTM autorregresiva.** Realimenta su predicción junto con el calendario de la hora siguiente.
- **Seq2seq.** Codificador LSTM; el decoder recibe el calendario futuro.

Los modelos de librería usan NeuralForecast. N-HiTS, TCN, iTransformer y DeepAR reciben el calendario futuro; PatchTST no admite exógenas. Todas las series entran con z-score por serie ajustado en train. El escalado por ventana de NeuralForecast divide por `eps` en las ventanas constantes de clientes con tramos en cero (por ejemplo MT_223) y producía pérdidas NaN.

### Tabla de resultados en prueba de las redes"""
        ),
        code(
            """columnas = ["modelo", "familia", "L", "estrategia", "usa_covariables", "total_parameters",
            "trainable_parameters", "training_seconds", "inference_seconds", "hardware",
            "validation_mase", "mae", "rmse", "wape", "mase", "mase_navidad", "pinball_loss", "cobertura_p10_p90"]
resultados.sort_values("mase")[[c for c in columnas if c in resultados]].round(4)"""
        ),
        code(
            """# Redes junto a los baselines, LightGBM y Chronos (mismo backtest de prueba).
otros = pd.read_csv(ARTEFACTOS / "resultados_modelos.csv")
comparacion = pd.concat([resultados.assign(grupo="red"), otros.assign(grupo="referencia")], ignore_index=True)
comparacion.sort_values("mase")[["modelo", "grupo", "familia", "mae", "wape", "mase", "training_seconds"]].round(4)"""
        ),
        md(_analysis("tabla")),
        md(
            """### 5.1 Estrategias multi-paso

Las tres variantes comparten la configuración base: `L = 168`, estado oculto 64, una capa y las mismas covariables. Solo cambia la estrategia:
- **Autorregresiva:** un paso a la vez, realimentando la predicción.
- **Multi-salida:** una capa densa devuelve las 24 horas.
- **Encoder-decoder:** el decoder recibe el calendario de las 24 horas futuras.

La comparación se hace en validación, para no usar prueba en una decisión de diseño, y se repite con los pronósticos de prueba de cada red."""
        ),
        code(
            """datos = PanelTensors(panel, "cuda" if torch.cuda.is_available() else "cpu")
origenes_validacion = daily_origins(CONFIG.validation_start, CONFIG.validation_end)
estrategias = {"LSTM-AR": "autorregresiva", "LSTM": "multi-salida", "Seq2Seq": "encoder-decoder"}
curvas = []
for config in base_configs():
    if config.label not in estrategias:
        continue
    corrida = load_run(ARTEFACTOS, run_id_of(config), datos)
    pronosticos = rolling_forecasts(
        lambda p, c: predict_positions(corrida.model, corrida.config, datos, corrida.normalizer, p, c),
        datos, origenes_validacion, config.label)
    pronosticos["error"] = (pronosticos["y"] - pronosticos["y_hat"]).abs()
    curva = pronosticos.groupby("horizon")["error"].mean().rename("mae").reset_index()
    curva["estrategia"] = estrategias[config.label]
    curvas.append(curva)
curvas = pd.concat(curvas)
curvas.to_csv(ARTEFACTOS / "detalle_redes" / "mae_horizonte_estrategias_validacion.csv", index=False)

prueba_horizonte = pd.read_csv(ARTEFACTOS / "detalle_redes" / "metricas_horizonte.csv")
fig, ejes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
for estrategia, grupo in curvas.groupby("estrategia"):
    ejes[0].plot(grupo["horizon"], grupo["mae"], marker="o", label=estrategia)
for modelo, estrategia in estrategias.items():
    grupo = prueba_horizonte.loc[prueba_horizonte["modelo"].eq(modelo)]
    ejes[1].plot(grupo["horizon"], grupo["mae"], marker="o", label=f"{estrategia} ({modelo})")
for eje, titulo in zip(ejes, ["Validación, configuración base", "Prueba, mejor configuración"]):
    eje.set_xlabel("Paso del horizonte h"); eje.set_ylabel("MAE (kWh)"); eje.set_title(titulo)
    eje.grid(alpha=0.25); eje.legend(frameon=False)
fig.tight_layout(); fig.savefig(ARTEFACTOS / "estrategias_multipaso.png", dpi=180, bbox_inches="tight")
curvas.pivot(index="horizon", columns="estrategia", values="mae").iloc[[0, 5, 11, 17, 23]].round(2)"""
        ),
        md(_analysis("5.1")),
        md(
            """### 5.2 Iteraciones de hiperparámetros

Cada iteración cambia un solo factor respecto a la configuración base. La LSTM se varía en `L`, meses de historia, normalización, estado oculto, capas y dropout, learning rate y batch; DLinear en `L`, historia y normalización; el MLP en tamaño, dropout y `L`.
- `iteraciones_entrenamiento.csv` tiene una fila por epoch: pérdidas, MAE y MASE de validación, parámetros, tiempos y memoria pico de GPU.
- `iteraciones_resumen.csv` resume cada corrida en su mejor epoch.

Todas las selecciones usan solo validación."""
        ),
        code(
            """propias = resumen.loc[resumen["kind"].isin(["mlp", "dlinear", "nlinear", "lstm", "lstm_ar", "seq2seq"])]
tabla_iter = propias[["modelo", "lookback", "history_months", "normalization", "hidden_size", "num_layers",
                      "dropout", "learning_rate", "batch_size", "best_epoch", "epochs_run", "validation_mae",
                      "validation_mase", "total_parameters", "epoch_seconds_mean", "training_seconds",
                      "inference_seconds", "peak_gpu_memory_mb"]]
tabla_iter.sort_values(["modelo", "validation_mase"]).round(4)"""
        ),
        code(
            """# Efecto de cada factor sobre la LSTM y DLinear (MASE de validación).
base = {c.label: c for c in base_configs()}
def efecto(modelo, columna):
    referencia = base[modelo]
    filas = propias.loc[propias["modelo"].eq(modelo)].copy()
    otros = {"lookback": referencia.lookback, "history_months": referencia.history_months,
             "normalization": referencia.normalization, "hidden_size": referencia.hidden_size,
             "num_layers": referencia.num_layers, "dropout": referencia.dropout,
             "learning_rate": referencia.learning_rate, "batch_size": referencia.batch_size}
    otros.pop(columna)
    if columna == "num_layers":
        otros.pop("dropout")
    for clave, valor in otros.items():
        filas = filas.loc[np.isclose(filas[clave], valor) if isinstance(valor, float) else filas[clave].eq(valor)]
    return filas[[columna, "validation_mase", "best_epoch", "training_seconds"]].sort_values(columna)

for modelo, factores in {"LSTM": ["lookback", "history_months", "normalization", "hidden_size", "num_layers",
                                  "learning_rate", "batch_size"],
                         "DLinear": ["lookback", "history_months", "normalization"]}.items():
    for factor in factores:
        print(f"{modelo} | {factor}"); print(efecto(modelo, factor).round(4).to_string(index=False)); print()"""
        ),
        code(
            """# Curvas de pérdida de varias iteraciones de la LSTM y del MLP.
seleccion = ["LSTM", "MLP", "DLinear"]
corridas = (propias.loc[propias["modelo"].isin(seleccion)]
            .sort_values("validation_mase").groupby("modelo").head(2)["run_id"].tolist())
corridas += propias.loc[propias["modelo"].eq("LSTM") & propias["normalization"].eq("global"), "run_id"].tolist()
fig, ejes = plt.subplots(1, 2, figsize=(13, 4.2))
for run_id in corridas:
    curva = iteraciones.loc[iteraciones["run_id"].eq(run_id)]
    etiqueta = f"{curva['modelo'].iloc[0]} | {curva['configuracion'].iloc[0].replace(' capas', chr(10) + 'capas')}"
    linea, = ejes[0].plot(curva["epoch"], curva["train_loss"], linestyle="--")
    ejes[0].plot(curva["epoch"], curva["validation_loss"], color=linea.get_color(), label=etiqueta)
    ejes[1].plot(curva["epoch"], curva["validation_mase"], color=linea.get_color(), marker=".")
ejes[0].set_title("Pérdida L1 normalizada (-- train, — validación)"); ejes[0].set_xlabel("epoch")
ejes[1].set_title("MASE de validación por epoch (escala log)"); ejes[1].set_xlabel("epoch"); ejes[1].set_yscale("log")
for eje in ejes: eje.grid(alpha=0.25)
fig.legend(loc="lower center", ncol=2, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.22))
fig.tight_layout(); fig.savefig(ARTEFACTOS / "curvas_perdida.png", dpi=180, bbox_inches="tight")"""
        ),
        code(
            """# ¿Menor pérdida de entrenamiento implica mejor MASE de validación?
finales = iteraciones.loc[iteraciones["validation_mase"].notna()].copy()
ultimo = finales.sort_values("epoch").groupby("run_id").tail(1).merge(
    propias[["run_id", "kind", "normalization"]], on="run_id")
comparables = ultimo.loc[ultimo["normalization"].eq("serie")]
print("Correlación de Spearman entre corridas (pérdida de train final vs MASE de validación):",
      round(comparables[["train_loss", "validation_mase"]].corr(method="spearman").iloc[0, 1], 3))
pares = []
for run_id, curva in finales.groupby("run_id"):
    curva = curva.sort_values("epoch")
    baja = curva["train_loss"].diff().lt(0)
    sube = curva["validation_mase"].diff().gt(0)
    pares.append({"run_id": run_id, "epochs": len(curva), "train_baja_y_val_sube": int((baja & sube).sum())})
pares = pd.DataFrame(pares)
print("Epochs en que la pérdida de train bajó pero el MASE de validación subió:",
      int(pares["train_baja_y_val_sube"].sum()), "de", int(pares["epochs"].sum() - len(pares)))"""
        ),
        md(_analysis("5.2")),
        md(
            """### Sensibilidad a la historia de entrenamiento

Gráfica 3 del laboratorio: MASE de validación con 3, 12 y 30 meses de historia antes del 1 de julio de 2014.
- **Modelos:** seasonal naive, LightGBM global, DLinear, la mejor recurrente según validación y PatchTST.
- **Referencia:** Chronos-Bolt zero-shot no se entrena, así que aparece como línea constante.
- **Escala MASE:** usa siempre toda la historia previa al origen, igual que `backtest`, para que los puntos sean comparables."""
        ),
        code(
            """sensibilidad = pd.read_csv(ARTEFACTOS / "sensibilidad_historia.csv")
plot_history_sensitivity(sensibilidad, ARTEFACTOS / "sensibilidad_historia.png")
sensibilidad.pivot(index="modelo", columns="history_months", values="mase").round(4)"""
        ),
        code(
            """from IPython.display import Image
Image(str(ARTEFACTOS / "sensibilidad_historia.png"))"""
        ),
        md(_analysis("historia")),
        md(
            """### Pronóstico probabilístico (DeepAR)

DeepAR se entrena con verosimilitud t de Student. De sus trayectorias se obtienen P10, P50 y P90. Se reporta:
- la pinball loss promedio en esos tres cuantiles;
- la cobertura empírica del intervalo P10-P90, cuyo valor nominal es 80 %."""
        ),
        code(
            """cuantiles = pd.read_csv(ARTEFACTOS / "cuantiles_deepar.csv", parse_dates=["ds"])
real = panel.rename_axis("ds").reset_index().melt(id_vars="ds", var_name="unique_id", value_name="y")
cuantiles = cuantiles.merge(real, on=["unique_id", "ds"])
cuantiles["dentro"] = cuantiles["y"].between(cuantiles["p10"], cuantiles["p90"])
print(resultados.loc[resultados["modelo"].eq("DeepAR"), ["pinball_loss", "cobertura_p10_p90", "mase"]].round(4).to_string(index=False))
por_hora = cuantiles.groupby(cuantiles["ds"].dt.hour)["dentro"].mean()
ventana = cuantiles.loc[cuantiles["unique_id"].eq("MT_001") & cuantiles["ds"].between("2014-12-20", "2014-12-31 23:00")]
fig, ejes = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw={"width_ratios": [2.2, 1]})
ejes[0].fill_between(ventana["ds"], ventana["p10"], ventana["p90"], alpha=0.3, label="P10-P90")
ejes[0].plot(ventana["ds"], ventana["p50"], label="P50"); ejes[0].plot(ventana["ds"], ventana["y"], color="black", lw=0.8, label="real")
ejes[0].set_title("DeepAR, MT_001, 20-31 de diciembre (kWh)"); ejes[0].legend(frameon=False)
ejes[0].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%d-%m"))
ejes[1].bar(por_hora.index, por_hora.values); ejes[1].axhline(0.8, color="black", ls="--", lw=0.8)
ejes[1].set_title("Cobertura P10-P90 por hora"); ejes[1].set_xlabel("hora")
fig.tight_layout(); fig.savefig(ARTEFACTOS / "intervalos_deepar.png", dpi=180, bbox_inches="tight")"""
        ),
        md(_analysis("deepar")),
        md(
            """### 5.3 Modelo local contra global

La mejor red propia según validación se entrena de dos formas:
- **Local:** un modelo por cliente, con la misma configuración y el mismo early stopping, solo con las ventanas de ese cliente.
- **Global:** el modelo ya entrenado con los 100 clientes.

Los 10 clientes se eligen con la semilla del experimento y el MASE se mide en prueba."""
        ),
        code(
            """local_global = pd.read_csv(ARTEFACTOS / "local_vs_global.csv")
print("Modelo base:", local_global["modelo_base"].iloc[0])
print("MASE medio local:", round(local_global["mase_local"].mean(), 4), "| global:", round(local_global["mase_global"].mean(), 4),
      "| clientes donde gana el global:", int(local_global["mejora_global"].gt(0).sum()), "de", len(local_global))
local_global.sort_values("mejora_global", ascending=False).round(4)"""
        ),
        md(_analysis("5.3")),
        md(
            """### Sensibilidad a la semilla y reentrenamiento mensual

- **Semillas:** la mejor red según validación se repite con tres semillas.
- **Reentrenamiento mensual:** la mejor red propia (LSTM) y la mejor red en general (N-HiTS) se reentrenan al inicio de octubre, noviembre y diciembre, con ventana expandible y el mismo presupuesto elegido en validación: epochs para la LSTM, pasos para N-HiTS. Cada modelo pronostica solo su mes. La fila «sin reentrenar» compara contra los pesos fijos al 30 de septiembre."""
        ),
        code(
            """semillas = pd.read_csv(ARTEFACTOS / "semillas.csv")
print(semillas.round(4).to_string(index=False))
print("Desviación estándar del MASE (validación, prueba):",
      round(semillas["validation_mase"].std(ddof=1), 4), round(semillas["test_mase"].std(ddof=1), 4))
mensual = pd.read_csv(ARTEFACTOS / "reentrenamiento_mensual_redes.csv")
mensual.round(4)"""
        ),
        md(_analysis("semillas")),
        md(
            """### Error por hora del día y semana navideña

Desgloses pedidos en la sección 4 para cada red: `artifacts/detalle_redes/metricas_{horizonte,hora,cliente,navidad}.csv`."""
        ),
        code(
            """por_hora = pd.read_csv(ARTEFACTOS / "detalle_redes" / "metricas_hora.csv")
navidad = pd.read_csv(ARTEFACTOS / "detalle_redes" / "metricas_navidad.csv", parse_dates=["ds"])
fig, ejes = plt.subplots(1, 2, figsize=(13, 4))
for eje in ejes: eje.set_prop_cycle(color=plt.cm.tab20.colors[::2] + plt.cm.tab20.colors[1::2])
for modelo, grupo in por_hora.groupby("modelo"):
    ejes[0].plot(grupo["hour"], grupo["mae"], label=modelo)
for modelo, grupo in navidad.groupby("modelo"):
    ejes[1].plot(grupo["ds"].dt.strftime("%d-%m"), grupo["mase"], marker="o", label=modelo)
ejes[0].set_title("MAE por hora del día (prueba)"); ejes[0].set_xlabel("hora"); ejes[0].set_ylabel("MAE (kWh)")
ejes[1].set_title("MASE diario, 22-28 de diciembre"); ejes[1].set_ylabel("MASE")
for eje in ejes: eje.grid(alpha=0.25)
ejes[1].legend(frameon=False, fontsize=7, ncol=2)
fig.tight_layout(); fig.savefig(ARTEFACTOS / "error_hora_navidad_redes.png", dpi=180, bbox_inches="tight")
navidad.pivot(index="ds", columns="modelo", values="mase").round(3)"""
        ),
        md(_analysis("navidad")),
        md(
            """### Observaciones para la discusión (sección 8)"""
            + ("\n\n" + _analysis("observaciones") if _analysis("observaciones") else "")
        ),
    ]
    notebook = nbf.v4.new_notebook()
    notebook.metadata.update(
        {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        }
    )
    notebook.cells = [cell for cell in cells if cell.source.strip()]
    destination = Path(output)
    nbf.write(notebook, destination)
    return destination


if __name__ == "__main__":
    build_notebook()
