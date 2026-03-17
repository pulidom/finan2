# Estrategias de Mean Reversion y Optimización de Portafolio
### Basado en Kakushadze (2014) y en nuestros experimentos de pairs trading


Las estrategias de **statistical arbitrage** buscan explotar desalineamientos temporales entre activos correlacionados.

Un ejemplo clásico es el **pairs trading**:

Si dos activos históricamente se mueven juntos y se separan temporalmente, se espera que el spread vuelva a su media.

Estrategia:

long activo barato  
short activo caro

# Pairs Trading y Cointegración

Si dos activos están cointegrados:

$$
Spread_t = Y_t - β X_t
$$

El spread oscila alrededor de una media.

Esto genera oportunidades de trading cuando el spread se desvía de esa media.

Señal típica:

$$
z_t = (spread_t - μ) / σ
$$

# Señales de Trading

En nuestro proyecto usamos el z-score del spread:

Si:

z > threshold → abrir posición short

z < -threshold → abrir posición long

En el código usamos por ejemplo:

sigma_co = 1.5   # abrir posición  
sigma_ve = 0.1   # cerrar posición

Estas señales aparecen frecuentemente cuando el spread presenta mean reversion

# Resultados

Se compararon dos métodos:

1) Moving Average
2) EWMA (media exponencial)

Y se evaluaron distintos pares cointegrados.

## Resultados del experimento (Moving Average)

| Par        | Capital final | Volatilidad |
|------------|--------------|-------------|
| EQT–CTRA   | 120.20       | 1.45        |
| DHT–STNG   | 120.40       | 4.06        |
| LTBR–CLNE  | 738.37       | 1.08        |
| STNG–DHT   | 161.58       | 0.46        |
| EQT–SM     | 5.20         | 4.04        |
| CLNE–SFL   | 64.94        | 1.04        |
| DHT–ASC    | 143.14       | 1.16        |
| TEN–STNG   | 168.09       | 3.77        |
| SM–EQT     | 37.88        | 3.54        |
| NOG–OXY    | 217.91       | 4.64        |

El par LTBR–CLNE:738.37 mostró el mejor desempeño.
En cambio EQT–SM prácticamente no generó ganancias. 

Ademas podemos ver que no siempre mayor volatilidad del spread implica mayor rentabilidad.


| Par        | Volatilidad | Capital |
|------------|-------------|---------|
| LTBR-CLNE  | 1.08        | 738     |
| NOG-OXY    | 4.64        | 217     |
| EQT-SM     | 4.03        | 5       |

**Insight clave:**

La rentabilidad depende no solo de la volatilidad,
sino también de la estabilidad de la reversión a la media.

## Resultados del experimento (EWMA)

| Par        | Capital final | Volatilidad |
|------------|--------------|-------------|
| EQT–CTRA   | 134.76       | 2.19        |
| DHT–STNG   | 172.29       | 10.74       |
| LTBR–CLNE  | 141.71       | 2.66        |
| STNG–DHT   | 110.01       | 1.43        |
| EQT–SM     | 24.88        | 7.43        |
| CLNE–SFL   | 35.43        | 1.91        |
| DHT–ASC    | 203.40       | 2.60        |
| TEN–STNG   | 296.07       | 10.35       |
| SM–EQT     | 48.29        | 5.94        |
| NOG–OXY    | 182.00       | 14.89       |

# Estimación de la Media del Spread

## Moving average produjo el mayor retorno individual 

El par: 

LTBR – CLNE 

Generó capital = 738 

muy superior al resto. 

Esto sugiere que la media móvil puede capturar movimientos de reversión más amplios. 


## EWMA produce resultados más balanceados 

Con el método exponencial: 

varios pares generan capital entre 100 y 300 

el mejor fue: 

TEN – STNG : 296 

Esto indica una estrategia más estable entre pares. 

 
## EWMA genera spreads más volátiles 

Las volatilidades aumentan bastante: 

DHT-STNG 
moving : 4.05 
exp : 10.74 

Esto ocurre porque EWMA responde más rápido a cambios recientes. 


## Resultados:

Moving average
• señales más estables
• menos frecuentes
• retornos individuales mayores

EWMA
• más sensible a cambios recientes
• más señales
• resultados más balanceados entre pares

# Conexión con el Paper de Kakushadze (2014)

El paper propone un framework para estrategias de mean reversion:

Pair trading <br>
↓ <br>
Demeaning returns <br>
↓ <br>
Regression <br>
↓ <br>
Weighted regression <br>
↓ <br>
Optimization <br>
↓ <br>
Factor models        

Es decir, las señales son solo el primer paso.

Luego hay que construir un portafolio óptimo.

# Mean Reversion

La idea básica de una estrategia de mean reversion es explotar desviaciones
temporales de activos que normalmente se mueven juntos.

Para dos activos A y B:

$$
RA = log(PA(t2)/PA(t1))
$$
$$
RB = log(PB(t2)/PB(t1))
$$

Definimos retornos demeaned:

$$ 
R̄ = (RA + RB) / 2
$$

$$
R̃A = RA − R̄
$$

$$
R̃B = RB − R̄
$$

Interpretación:

$R̃ > 0  → activo\ caro → short  $
$R̃ < 0  → activo\ barato → long$

# Mean Reversion vía Regresión

Para N activos:

$$
R_i = Σ Λ_{iA} f_A + ε_i
$$

donde

$R_i$ = retornos  
$Λ_{iA}$ = matriz de loadings (por ejemplo industria)  
$f_A$ = factores  
$\epsilon_i$ = residuales de la regresión

La señal de trading pasa a ser: $ε_i$

Los residuales capturan desviaciones respecto al comportamiento esperado.
Esto conecta directamente con el concepto de spread que usamos en pairs trading.

# Weighted Regression

Pesos de regresión:

$$
z_i = 1 / {σ_i}²
$$

Entonces:

$$
\epsilon = R - \Omega (\Omega^T Z \Omega)^{-1} \Omega^T Z R
$$

Esto reduce la exposición a activos muy volátiles.


# Problema de Optimización de Portafolio

Una vez generadas las señales, debemos decidir cuánto invertir en cada activo.

El problema típico es maximizar el **Sharpe ratio**:

$$
S = P / V
$$

donde

$P = Σ R_i D_i$

$V = √( Σ C_{ij} D_i D_j )$

El problema puede verse como:

$\text{max }   (wᵀR / √(wᵀ C w))$

Donde podemos sacar que la solucion sin restricciones es: 

$$
w = C⁻¹ R
$$

donde

w = pesos del portafolio  
R = retornos esperados (señales)  
C = matriz de covarianza

# Restricciones del Portafolio

En estrategias long-short se suelen imponer restricciones:

dollar neutrality

$∑ w_i = 0$

Donde los límites de posición se da en 

|w_i| ≤ límite

Estas restricciones ayudan a controlar el riesgo del portafolio.


# Modelos Factoriales de Riesgo

Kakushadze propone usar modelos factoriales:

$$
Θ = Ξ + Ω Φ Ωᵀ
$$

donde

$Ξ$  = riesgo específico  
$Ω$  = loadings de factores  
$Φ$  = covarianza entre factores

Esto reduce la dimensionalidad del problema y produce estimaciones más estables.

# Trabajo futuro

Actualmente evaluamos pares de forma independiente.

Una extensión natural sería:

- optimizar un portafolio conjunto de pares en lugar de evaluarlos de forma independiente
- incorporar modelos factoriales para estimar el riesgo
- analizar la estabilidad temporal de las señales

Esto permitiría pasar de un enfoque descriptivo a uno completamente cuantitativo.

El objetivo es pasar de:

pairs trading independientes

a

un portafolio optimizado de estrategias de mean reversion 

# Conclusión

La estrategia del paper completa puede verse como:

señales de mean reversion <br>
↓<br>
estimación de retornos<br>
↓<br>
modelo de riesgo<br>
↓<br>
optimización de portafolio

El paper de Kakushadze muestra que regresión y optimización pueden verse como partes de un mismo framework cuantitativo.