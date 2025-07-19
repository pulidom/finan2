# Experimentos Eze
Este repositorio es un clonado del de https://github.com/pulidom/finan2 con los experimentos y cambios que realicé.


# primeras_pruebas_hiperp
Aqui se encuentra una exploracion preliminar de los hiperparametros para tratar de optimizarlos seleccionando los pares por pval o capital y en algunos casos como en la carpeta 
pruebas_nsharp_positive_days probando seleccionando por sharpe o por cantidad de dias positivos. Se fue evaluando la sensibilidad a Ntrain (periodo de seleccion de pares para los Njumps dias siguientes),
Njump (dias durante los que se utilizan los pares seleccionados en el ultimo Ntrain), sigma_co (umbral de Z-score de compra), sigma_ve (umbral de Z-score de venta)

# best_hip
Aqui se encuentran ya en una evaluacion mas robusta para 3 diferentes sectores la sensibilidad a la variacion de los mismos hiperparametros tanto seleccionando los pares por pval como
por el half life utilizado (seleccion de dias de menor half life pero con hl >3 para evitar ruido). Se grafica el capital final para cada combinacion de hiperparametros, el sharpe ratio 
calculado como la media de retornos sobre el desvio y la mediana

# ultimas_pruebas
Dentro de test_industrias se encuentra una evaluacion de hiperparametros para cada sector para verificar como impactan en cada uno de los sectores por separado utilizando half life para seleccion y calculando el sharpe ratio  tanto como se explico en best_hip como mediante la funcion de cointegracion. Tambien se grafican los retornos y el capital a lo largo de las series, solo se evaluo la sensibilidad al Ntrain
En pruebaBioMachMedia_Nmax-1 se encuentra la seleccion de pares utilizando esos tres sectores juntos
En prueba_all_Nmax20 se prueba una seleccion utilizando todos los sectores pero utilizando solo las primeras 20 series de activos de cada sector

# retornos_paralel.py
En este codigo se utiliza el modelo para calcular las ganancias para cada configuracion de hiperparametros mediante paralelizacion. Se guardan distintas variables en distintos dataframes
que despues se grafican con los distintos codigos de plot

# test_industrias.py
Algo similar especializando industria por industria

