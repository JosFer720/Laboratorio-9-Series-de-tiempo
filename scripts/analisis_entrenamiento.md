# Análisis de la sección 5

Cada bloque se inserta en `entrenamiento.ipynb` debajo de la celda correspondiente
(ver `scripts/build_entrenamiento_notebook.py`).

<!-- SECCION: tabla -->
**Lectura de la tabla.**

**Ranking en prueba (MASE).**
- N-HiTS (0,770) es la mejor red. Le siguen iTransformer (0,785) y la LSTM multi-salida con estado oculto 128 (0,799).
- Las tres superan a Chronos-Bolt zero-shot (0,809), el mejor modelo de referencia.
- Le siguen, con MASE entre 0,82 y 0,85: MLP (0,824), TCN (0,831), Seq2Seq (0,831), PatchTST (0,835), LSTM-AR (0,838) y DLinear (0,851).
- NLinear (0,874) queda algo detrás.
- Todas las redes deterministas superan a SeasonalNaive24 (1,024), AutoETS (1,025) y LightGBM (1,693).
- DeepAR (1,249) es la única red peor que el seasonal naive.

**Validación y prueba.** El orden en validación coincide con el de prueba en los extremos: N-HiTS es primero y DeepAR último en ambos. En prueba todos los MASE bajan entre 0,01 y 0,06. El trimestre octubre-diciembre resulta, en promedio, más fácil que julio-septiembre para la escala MASE.

**Costo.**
- DLinear, NLinear y el MLP entrenan en menos de 1 minuto con menos de 1 GB de GPU.
- Las LSTM y los modelos de NeuralForecast tardan de 1 a 15 minutos. La excepción es DeepAR, con 48 minutos compartiendo GPU.
- La inferencia de prueba (~600 s por modelo) la domina el bucle de pandas de `backtest`, no la red.

**Advertencia sobre tiempos.** Varias corridas compartieron la GPU con otro proceso, así que su `training_seconds` es una cota superior:
- LSTM base, LSTM-AR, Seq2Seq, las variantes de `L`, historia, normalización y estado oculto 32 de la LSTM.
- TCN, PatchTST, iTransformer y DeepAR.
- La corrida de N-HiTS de la tabla.

Como referencia de tiempo sin compartir, N-HiTS tarda 63-65 s (corridas de semillas) frente a 244 s en la tabla, y la LSTM base tarda 9 s por epoch (variantes de learning rate) frente a 38 s.

<!-- SECCION: 5.1 -->
**Estrategias multi-paso.**

**Validación, configuración base.**
- El encoder-decoder es el mejor (MASE 0,876), seguido de la autorregresiva (0,878) y la multi-salida (0,889). La diferencia total es de 1,5 %.
- Las curvas por horizonte muestran dónde está la diferencia, no solo cuánto es.

**Primeras horas y rampa de la mañana.** Los orígenes son a medianoche, así que el paso `h` equivale a la hora `h-1`.
- Entre `h = 4` y `h = 7` (madrugada), la autorregresiva y el encoder-decoder tienen MAE de ~25-33 kWh, contra 40-50 de la multi-salida.
- En la rampa de la mañana (`h = 9-10`), la multi-salida tiene el pico más alto (173 kWh) y el encoder-decoder el más bajo (74 y 124 kWh).
- Conocer el calendario de cada hora futura dentro del decoder ayuda a ubicar el cambio de régimen.

**Final del horizonte.**
- En `h = 23-24` la autorregresiva acumula error (114-115 kWh), mientras la multi-salida baja a 96.
- Es el patrón esperado: la autorregresiva realimenta sus propias predicciones y el error se compone; la multi-salida predice cada paso directamente.

**Prueba, mejor configuración de cada una.**
- La multi-salida elegida tiene estado oculto 128, así que gana (0,799 contra 0,831 y 0,838). Esa comparación mezcla capacidad y estrategia.
- Aun así se repite el patrón: la multi-salida es claramente mejor en `h = 1` y `h = 24`, y la autorregresiva es mejor en `h = 7-8`.

**Costo.**
- La autorregresiva ejecuta 24 pasos secuenciales de la LSTM por ventana. Es la más lenta de entrenar e inferir (~53 s por epoch compartiendo GPU, contra ~11 s del encoder-decoder).
- La multi-salida es la más barata.

**Conclusión.** Con el mismo tamaño, el encoder-decoder con calendario futuro es la estrategia más precisa. La multi-salida es la de mejor relación costo-precisión, y la que mejor escala en capacidad.

<!-- SECCION: 5.2 -->
**Iteraciones.** Se cambia un factor a la vez sobre la LSTM base (MASE de validación 0,889) y DLinear (0,886).

**Normalización: el factor más importante.**
- La normalización global (una sola media y desviación para todos los clientes) lleva el MASE a 1,46 en la LSTM y a 1,59 en DLinear.
- El consumo medio va de ~5 kWh (MT_001) a ~49 000 kWh (MT_362), así que la pérdida queda dominada por los clientes grandes y los pequeños se ignoran.
- La normalización por ventana (estilo RevIN) mejora un poco a la por serie: 0,868 contra 0,889 en la LSTM, 0,880 contra 0,886 en DLinear.

**Longitud de contexto.**
- `L = 48` empeora a ambos (0,908 y 0,945) porque no ve el ciclo semanal.
- `L = 336` mejora un poco (0,884 y 0,873); es la mejor configuración de DLinear y del MLP.

**Historia.** La LSTM depende mucho de la cantidad de datos (1,341 con 3 meses, 0,965 con 12, 0,889 con 30). DLinear es insensible.

**Capacidad.**
- Con estado oculto 32 / 64 / 128, la LSTM da 0,948 / 0,889 / 0,827: el factor con mayor mejora.
- Dos capas dan 0,847. Dropout 0,2 entre capas no ayuda (0,850): no hay sobreajuste dentro del presupuesto.

**Optimización.**
- Con learning rate 3e-4 / 1e-3 / 3e-3, el MASE es 0,982 / 0,889 / 0,835.
- Con batch 256 / 1024 / 4096, el MASE es 0,834 / 0,889 / 0,994.
- En este rango, más pasos de gradiente o pasos más grandes ayudan. Esto, y que casi todas las LSTM tengan su mejor epoch en 22-25 de un máximo de 25, indica que las LSTM terminaron **subentrenadas**: el tope de 25 epochs las limita y sus diferencias reflejan en parte el presupuesto.
- DLinear, NLinear y el MLP convergen en 3-12 epochs y paran por early stopping.

**MLP.** Con 64 unidades sin dropout (0,848) empata con 256 con dropout 0,1 (0,853). La capacidad extra no aporta.

**¿Menor pérdida de entrenamiento implica mejor MASE de validación?**
- **Entre configuraciones** con la misma normalización, sí en promedio: la correlación de Spearman entre la pérdida de train final y el MASE de validación es 0,85.
- **Dentro de una corrida, no siempre.** En 135 de 522 transiciones entre epochs (26 %) la pérdida de train bajó mientras el MASE de validación subió. Pasa sobre todo en los modelos lineales y el MLP, que se ajustan en pocas epochs.
- **El contraejemplo más claro es la normalización global.** Tiene la pérdida de train más baja de todas las corridas (~0,016) y el peor MASE (1,46). La pérdida está en otra escala y mide el ajuste a los clientes grandes.
- La pérdida solo compara corridas con la misma escala. La selección debe hacerse con la métrica de validación en kWh o MASE.

**Memoria.** El pico de GPU de la LSTM no cambia con el batch (8,57 GB con 256 y con 4096), así que no lo fija el lote de entrenamiento. Con `L = 336` o estado oculto 128 sube a ~17 GB, más que los 16 GB de la tarjeta, y Windows recurre a memoria compartida del sistema.

<!-- SECCION: historia -->
**Sensibilidad a la historia.** MASE de validación con 3, 12 y 30 meses de entrenamiento.

**Los modelos se separan en tres grupos.**
- **Casi insensibles:** DLinear (0,886 / 0,882 / 0,886) y PatchTST (0,911 / 0,882 / 0,891). Con 3 meses ya tienen ~2 000 ventanas por cliente para sus pocos parámetros efectivos. PatchTST incluso es un poco mejor con 12 meses que con 30, aunque la diferencia es de 0,01.
- **Muy dependiente:** Seq2Seq (1,666 / 0,928 / 0,876). Con 3 meses no aprende la estacionalidad semanal ni la diferencia entre temporadas; el early stopping corta en la epoch 7. Necesita al menos un año.
- **No monótono:** LightGBM (1,309 / 1,271 / 1,382). Su recursión hora a hora acumula error en cualquier caso. Con 30 meses incluye 2012, y una explicación posible es que esos niveles de consumo se parecen menos a los de 2014.

**Referencias constantes.**
- Chronos-Bolt zero-shot (0,773) es mejor que cualquier modelo entrenado en validación; N-HiTS, el mejor entrenado, da 0,794.
- SeasonalNaive24 (0,994) queda por encima de todas las redes con 12 o más meses.
- Con poca historia (3 meses), Chronos y los modelos lineales son la opción segura. Con 30 meses, las redes más grandes (N-HiTS, LSTM con estado oculto 128) alcanzan o superan a Chronos en prueba.

<!-- SECCION: deepar -->
**Pronóstico probabilístico.**
- **Pinball loss** (promedio de P10, P50 y P90): 33,8 kWh.
- **Cobertura del intervalo P10-P90:** 61 % contra el 80 % nominal. Los intervalos son demasiado estrechos, es decir, el modelo está sobreconfiado.

**Cobertura por hora.**
- Solo llega al 80 % a medianoche (`h = 1`). Hay un segundo máximo entre las 6 h y las 8 h (~69-77 %).
- Baja a ~50 % entre las 14 h y las 19 h, justo donde el error absoluto es mayor.
- Esto sugiere que la dispersión de la t de Student, ajustada en escala z por serie, no crece lo suficiente con el horizonte ni con el nivel de consumo diurno.

**Pronóstico puntual.**
- La mediana de DeepAR (MASE 1,249) es peor que SeasonalNaive24.
- Muestra un error grande en la rampa de las 7-8 h (MAE de 171 kWh a las 8 h). El muestreo autorregresivo propaga el error del inicio del día.

**Posibles mejoras, no exploradas aquí:**
- más pasos de entrenamiento o early stopping con pinball en lugar de la verosimilitud;
- verosimilitud por cuantiles;
- calibración conformal de los intervalos con los residuos de validación.

<!-- SECCION: 5.3 -->
**Local contra global.**
- **Configuración:** la mejor red propia, la LSTM multi-salida con estado oculto 128. Para cada uno de los 10 clientes se entrenó un modelo solo con sus ventanas, con la misma configuración y early stopping.
- **Resultado:** el modelo global gana en los 10 clientes. El MASE medio de prueba es 0,823 global contra 0,967 local, 15 % menos error.

**Dónde gana más el global.**
- En los clientes chicos o irregulares: MT_066 (37,5 kWh medios) pasa de 1,558 local a 1,124 global, y MT_225 de 0,881 a 0,649.
- En esos casos el modelo local para en 3-4 epochs: con ~21 000 ventanas muy correlacionadas, 74 000 parámetros se sobreajustan enseguida.
- Compartir la red entre clientes funciona como aumento de datos y regularización: aprende la forma diaria y semanal común y se adapta a cada nivel por la normalización por serie.

**Costo.** Un modelo local tarda 27-96 s. Con 100 clientes serían ~1,5-2 h y 100 conjuntos de pesos, contra ~11 min y un solo modelo para el global.

<!-- SECCION: semillas -->
**Semillas.**
- N-HiTS repetido con las semillas 3092, 1 y 2 da MASE de prueba de 0,770, 0,767 y 0,769.
- La desviación estándar es 0,0017 en prueba y 0,0016 en validación.
- La diferencia con iTransformer (0,015) es unas 9 desviaciones. Contra Chronos (0,04) y la LSTM (0,03) es aún mayor. Esas brechas no son ruido de inicialización.
- Ese cálculo vale para N-HiTS, que converge con early stopping. Las LSTM terminaron subentrenadas y pueden tener más varianza entre semillas.
- La semilla 3092 reproduce exactamente la corrida original: el MASE de prueba del evaluador vectorizado coincide con el de `backtest` hasta el sexto decimal.

**Reentrenamiento mensual** con ventana expandible de la mejor red propia (LSTM con estado oculto 128, 22 epochs fijas) y de la mejor red general (N-HiTS, con los pasos de su corrida original y sin early stopping):

| Mes | LSTM reentrenando | N-HiTS reentrenando |
|---|---|---|
| Octubre | 0,752 | 0,738 |
| Noviembre | 0,692 | 0,679 |
| Diciembre | 0,926 | 0,893 |
| Total | 0,791 | 0,771 |
| Sin reentrenar | 0,799 | 0,770 |

- La LSTM mejora 1,1 % (MAE de 68,0 a 66,5 kWh) y N-HiTS no mejora (0,771 frente a 0,770, MAE de 66,8 a 68,1 kWh). El patrón del panel cambia poco en tres meses, y el modelo sin reentrenar ya incluye la normalización por serie.
- El seasonal naive mensual de referencia da 0,948, 0,904 y 1,216. La red mantiene una ventaja de ~20-25 % en cada mes.
- En operación, reentrenar mensualmente cuesta tres entrenamientos de ~10 min y aporta poco. Bastaría con reentrenar trimestralmente o ante cambios detectados en el error.

<!-- SECCION: navidad -->
**Hora del día y semana navideña.**

**Hora del día.**
- El error de todas las redes sigue el perfil de consumo: ~25-35 kWh de MAE entre las 3 h y las 6 h, y ~85-110 kWh entre las 10 h y las 17 h.
- El primer paso (hora 0) no es el más fácil: a medianoche varios clientes cambian de régimen. En las horas de oficina el nivel y la variabilidad son mayores.

**Semana navideña** (MASE diario del 22 al 28 de diciembre).
- Todas las redes empeoran el 24 (~1,6) y sobre todo el 25 (2,7 a 4,4).
- **El 25 de diciembre** está marcado como feriado en las covariables:
  - Los modelos que usan calendario lo aprovechan en distinta medida: iTransformer (2,74), DeepAR (3,04), LSTM (3,22) y Seq2Seq (3,28) quedan muy por debajo de los univariados DLinear y NLinear (4,42) y PatchTST (4,24).
  - El MLP (4,23) y N-HiTS (3,95) reciben el mismo indicador pero lo aprovechan menos. Con pocos feriados en train, el indicador pesa poco.
- **El 26 de diciembre** separa a los modelos que copian el día anterior: DLinear (2,99), NLinear (2,85) y DeepAR (4,28) arrastran el patrón del feriado. N-HiTS (0,93), iTransformer (1,03) y PatchTST (1,05) se recuperan de inmediato.
- En el agregado de la semana (`mase_navidad`), iTransformer es el mejor (1,24), seguido de N-HiTS (1,40) y Seq2Seq (1,42).

<!-- SECCION: observaciones -->
1. **Ranking.** Con el mismo panel y protocolo, las redes globales bien configuradas superan a todos los baselines y a LightGBM. Las tres mejores (N-HiTS 0,770, iTransformer 0,785, LSTM con estado oculto 128 0,799) también superan a Chronos zero-shot (0,809) en prueba. En validación, Chronos sigue siendo el mejor (0,773), así que la ventaja de las redes es pequeña y depende del periodo.
2. **Normalizar por serie es indispensable.** La normalización global multiplica el MASE por 1,6-1,8. La por ventana aporta otra mejora pequeña.
3. **Escalado de NeuralForecast.** Su escalado por ventana divide por `eps` en las ventanas constantes de clientes con tramos en cero (por ejemplo MT_223) y produce pérdidas NaN. Se resolvió con un z-score por serie ajustado en train y `scaler_type="identity"`.
4. **Lineales contra recurrentes.** Los modelos lineales (DLinear y NLinear) son el piso barato: 0,85-0,87 en menos de 30 s de entrenamiento. Las recurrentes solo los superan con más capacidad y más epochs.
5. **Las LSTM terminaron subentrenadas.** Mejor epoch en 22-25 de 25 posibles, y mejoran con learning rate más alto o batch más pequeño. Un presupuesto mayor probablemente cerraría parte de la brecha con N-HiTS.
6. **Estrategias multi-paso.** Con el mismo tamaño, el encoder-decoder con calendario futuro es el más preciso. La autorregresiva acumula error al final del día y es la más lenta. La multi-salida es la más eficiente y la mejor al escalar.
7. **Modelo global contra local.** El global gana en 10 de 10 clientes (−15 % de MASE), sobre todo en los clientes chicos, donde un modelo local se sobreajusta.
8. **Historia.** Las redes recurrentes necesitan al menos 12 meses. Los lineales y PatchTST son estables con 3. LightGBM no mejora con más historia.
9. **Sensibilidad a la semilla.** La desviación estándar del MASE de prueba de N-HiTS es 0,0017, mucho menor que las diferencias entre los primeros puestos.
10. **Reentrenamiento mensual.** Mejora apenas 1 % a la LSTM y nada a N-HiTS. No justifica su costo frente a un reentrenamiento menos frecuente.
11. **DeepAR.** Queda sobreconfiado (cobertura P10-P90 de 61 % contra 80 % nominal) y su mediana es peor que el seasonal naive. Habría que calibrar sus intervalos antes de usarlos.
12. **Navidad.** Es el punto débil común. El indicador de feriado ayuda el 25, sobre todo a iTransformer y a las recurrentes. Los modelos que copian el día anterior fallan además el 26.
13. **Mediciones de tiempo.** Varias corridas compartieron la GPU y sus tiempos son cotas superiores. Los tiempos de inferencia de prueba están dominados por `backtest` (~10 min por modelo) y no reflejan el costo de la red.
