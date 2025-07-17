# Experimentos Gaston

Este repositorio es un clonado del de [https://github.com/pulidom/finan2](https://github.com/pulidom/finan2) con los experimentos que realicé.

A continuación un resumen de los experimentos, cuyos outputs encontrarán en la carpeta tmp, en el orden en el cual se fueron realizando.

## metric_test_sector.py

Testeo de distintas métricas para selección de pares en distintos sectores.

Opera como main_operativo, es decir:
1. Calcula las métricas durante Ntraining días.
1. Selecciona los 5-10-20 mejores pares.
1. Invierte durante Njump días
1. Repite

**Notas:**
- La selección mediante _Johansen_ no está del todo "curada" (creo que multiplicándolo por -1 se arreglaría)
- No se están asignando pesos según performance

**Resultados:**
- La vida media (*half_life*) parece ser mejor salvo ciertos casos que compite con *p_values*.
- A veces el _Hurst_ se mete en la disputa pero no mucho más.
- Los sectores donde se obtuvieron mejores retornos al final de la simulación son:
   - biotechnology
   - machinery
   - marine
   - media
   - metals
   - oil
- Parece haber problemas con:
   - hotels

## metric_test_sector.py

Testeo de distintas métricas para selección de pares en distintos sectores con una asignación de pesos durante
la inversión. Los sectores donde se testeó son aquellos sectores donde se obtuvieron buenos resultados en el
experimento anterior

Opera como un *main_operativo* modificado, es decir:
1. Calcula las métricas durante Ntraining días.
1. Selecciona los 5-10-20 mejores pares.
1. Invierte durante Njump días. Asignando pesos a los 5-10-20 pares seleccionados.
   - La asignación de pesos se hace utilizando como parámetro la misma métrica
     que se utilizó para la selección de pares, no el rendimiento, no la
     volatilidad, no el volumen. Es decir que si la selección se hizo
     utilizando *p_val*, el peso se asigna con *p_val*.
1. Repite

**Nota:** 
- Cuando se seleccionan pares según el *half_life* se está seleccionando los pares con **menor** vida media.
  Para que se seleccione el half_life dentro de cierto intervalo debería usarse el *half_life_penalty*
  (implementado dentro de *cointegration_score*, dentro de *cointegratoin.py*). (el *half_life_penalty*, además,
  está siendo revisado por Ezequiel).
- Utilizar el *half_life* para asignar pesos puede ser un equivalente a utilizar la volatilidad para asignar pesos.

**Resultado:**
- Asignarle pesos a cada par no parece dar mucha ventaja con respecto a no asignar pesos cuando se obliga al sistema
  a elegir **n** pares... habría que revisar...



## weights_test_all_sectors.py

Ídem que el experimento anterior, pero ahora se intenta desligarse de los sectores. Para hacer eso tuvo que  realizar
una modificación en cómo se leían los datos así que encontrarán cómo lo hago en la última función ue aparece en _read_data.py_.

A partir de aquí, opera como un *main_operativo* modificado, es decir:
1. Calcula las métricas durante Ntraining días.
1. Selecciona los 5-10-20 mejores pares.
1. Invierte durante Njump días. Asignando pesos a los 5-10-20 pares seleccionados.
   - La asignación de pesos se hace utilizando como parámetro la misma métrica
     que se utilizó para la selección de pares, no el rendimiento, no la
     volatilidad, no el volumen. Es decir que si la selección se hizo
     utilizando p_values, el peso se asigna con p_values.
1. Repite

**Resultado:**
- HF despunta de sobremanera al seleccionar los pares con MENOR vida media.
- Creo que al asignarle peso según HF de cierto modo le estamos metiendo un
  input sobre la volatilidad. No estoy seguro de si es bueno o malo.
- Debería revisarse la lógica de todo... despunta demasiado
