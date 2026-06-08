# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "15b10a14-49d8-4232-afca-2659551cd285",
# META       "default_lakehouse_name": "Data_LakeHouse",
# META       "default_lakehouse_workspace_id": "981be398-56d1-4882-a799-8c7592882837",
# META       "known_lakehouses": [
# META         {
# META           "id": "15b10a14-49d8-4232-afca-2659551cd285"
# META         }
# META       ]
# META     }
# META   }
# META }

# MARKDOWN ********************

# # CREACIÓN TABLA SILVER DE LAS ENCUESTAS DE LOS AÑOS 2018, 2019, 2022 Y 2023

# CELL ********************

import pandas as pd
import numpy as np
import io

# 1. El mapa base con las 86 variables estandarizadas
map_data = """Año	Variable	Pregunta
2018	P_45	ultimo NIVEL de estudio aprobado (titulo)
2019	P_45	ultimo NIVEL de estudio aprobado (titulo)
2022	p_045	ultimo NIVEL de estudio aprobado (titulo)
2023	P_238	ultimo NIVEL de estudio aprobado (titulo)
2018	P_35	Sabe leer y escribir mas de un parrafo?
2019	P_35	Sabe leer y escribir mas de un parrafo?
2022	p_035	Sabe leer y escribir mas de un parrafo?
2023	P_233	Sabe leer y escribir mas de un parrafo?
2018	P_127	¿Las personas de este hogar están inscritas en PROGRAMAS LÚDICOS?
2019	P_127	¿Las personas de este hogar están inscritas en PROGRAMAS LÚDICOS?
2022	p_127	¿Las personas de este hogar están inscritas en PROGRAMAS LÚDICOS?
2023	P_257	¿Las personas de este hogar están inscritas en PROGRAMAS LÚDICOS?
2018	P_128	¿Las personas de este hogar están inscritas en PROGRAMAS RECREATIVOS?
2019	P_128	¿Las personas de este hogar están inscritas en PROGRAMAS RECREATIVOS?
2022	p_128	¿Las personas de este hogar están inscritas en PROGRAMAS RECREATIVOS?
2023	P_258	¿Las personas de este hogar están inscritas en PROGRAMAS RECREATIVOS?
2018	P_129	¿Las personas de este hogar están inscritas en PROGRAMAS DEPORTIVOS?
2019	P_129	¿Las personas de este hogar están inscritas en PROGRAMAS DEPORTIVOS?
2022	p_129	¿Las personas de este hogar están inscritas en PROGRAMAS DEPORTIVOS?
2023	P_259	¿Las personas de este hogar están inscritas en PROGRAMAS DEPORTIVOS?
2018	P_158	La unidad de vivienda cuenta con servicios publicos de: ENERGiA
2019	P_158	La unidad de vivienda cuenta con servicios publicos de: ENERGiA
2022	p_158	La unidad de vivienda cuenta con servicios publicos de: ENERGiA
2023	P_029	La unidad de vivienda cuenta con servicios publicos de: ENERGiA
2018	P_162	La unidad de vivienda cuenta con servicios publicos de: ACUEDUCTO
2019	P_162	La unidad de vivienda cuenta con servicios publicos de: ACUEDUCTO
2022	p_162	La unidad de vivienda cuenta con servicios publicos de: ACUEDUCTO
2023	P_033	La unidad de vivienda cuenta con servicios publicos de: ACUEDUCTO
2018	P_165	La unidad de vivienda cuenta con servicios publicos de: ALCANTARILLADO
2019	P_165	La unidad de vivienda cuenta con servicios publicos de: ALCANTARILLADO
2022	p_165	La unidad de vivienda cuenta con servicios publicos de: ALCANTARILLADO
2023	P_036	La unidad de vivienda cuenta con servicios publicos de: ALCANTARILLADO
2018	P_167	La unidad de vivienda cuenta con servicios públicos de Telefono (línea fija)
2019	P_167	La unidad de vivienda cuenta con servicios públicos de Telefono (línea fija)
2022	p_167	La unidad de vivienda cuenta con servicios públicos de Telefono (línea fija)
2023	P_038	La unidad de vivienda cuenta con servicios públicos de Telefono (línea fija)
2018	P_171	La unidad de vivienda cuenta con servicios publicos de: Gas natural (red)
2019	P_171	La unidad de vivienda cuenta con servicios publicos de: Gas natural (red)
2022	p_171	La unidad de vivienda cuenta con servicios publicos de: Gas natural (red)
2023	P_042	La unidad de vivienda cuenta con servicios publicos de: Gas natural (red)
2018	P_174	La unidad de vivienda cuenta con servicios públicos de Aseo (recolección)
2019	P_174	La unidad de vivienda cuenta con servicios públicos de Aseo (recolección)
2022	p_174	La unidad de vivienda cuenta con servicios públicos de Aseo (recolección)
2023	P_045	La unidad de vivienda cuenta con servicios públicos de Aseo (recolección)
2018	P_178	La unidad de vivienda cuenta con servicios publicos de: CONEXION A INTERNET
2019	P_178	La unidad de vivienda cuenta con servicios publicos de: CONEXION A INTERNET
2022	p_178	La unidad de vivienda cuenta con servicios publicos de: CONEXION A INTERNET
2023	P_049	La unidad de vivienda cuenta con servicios publicos de: CONEXION A INTERNET
2018	P_176	La unidad de vivienda cuenta con servicios públicos de Gas en pipeta
2019	P_176	La unidad de vivienda cuenta con servicios públicos de Gas en pipeta
2022	p_176	La unidad de vivienda cuenta con servicios públicos de Gas en pipeta
2023	P_047	La unidad de vivienda cuenta con servicios públicos de Gas en pipeta
2018	P_159	ENERGiA - CALIDAD
2019	P_159	ENERGiA - CALIDAD
2022	p_159	ENERGiA - CALIDAD
2023	P_030	ENERGiA - CALIDAD
2018	P_163	ACUEDUCTO - CALIDAD
2019	P_163	ACUEDUCTO - CALIDAD
2022	p_163	ACUEDUCTO - CALIDAD
2023	P_034	ACUEDUCTO - CALIDAD
2018	P_166	ALCANTARILLADO - CALIDAD
2019	P_166	ALCANTARILLADO - CALIDAD
2022	p_166	ALCANTARILLADO - CALIDAD
2023	P_037	ALCANTARILLADO - CALIDAD
2018	P_168	TELEFONO - CALIDAD
2019	P_168	TELEFONO - CALIDAD
2022	p_168	TELEFONO - CALIDAD
2023	P_039	TELEFONO - CALIDAD
2018	P_172	GAS NATURAL - CALIDAD
2019	P_172	GAS NATURAL - CALIDAD
2022	p_172	GAS NATURAL - CALIDAD
2023	P_043	GAS NATURAL - CALIDAD
2018	P_175	ASEO (Recoleccion) - CALIDAD
2019	P_175	ASEO (Recoleccion) - CALIDAD
2022	p_175	ASEO (Recoleccion) - CALIDAD
2023	P_046	ASEO (Recoleccion) - CALIDAD
2018	P_177	GAS EN PIPETA - CALIDAD
2019	P_177	GAS EN PIPETA - CALIDAD
2022	p_177	GAS EN PIPETA - CALIDAD
2023	P_048	GAS EN PIPETA - CALIDAD
2018	P_179	INTERNET - CALIDAD
2019	P_179	INTERNET - CALIDAD
2022	p_179	INTERNET - CALIDAD
2023	P_050	INTERNET - CALIDAD
2018	P_160	ENERGiA - SUSPENDIDO
2019	P_160	ENERGiA - SUSPENDIDO
2022	p_160	ENERGiA - SUSPENDIDO
2023	P_031	ENERGiA - SUSPENDIDO
2018	P_164	ACUEDUCTO - SUSPENDIDO
2019	P_164	ACUEDUCTO - SUSPENDIDO
2022	p_164	ACUEDUCTO - SUSPENDIDO
2023	P_035	ACUEDUCTO - SUSPENDIDO
2018	P_169	TELEFONO - SUSPENDIDO
2019	P_169	TELEFONO - SUSPENDIDO
2022	p_169	TELEFONO - SUSPENDIDO
2023	P_040	TELEFONO - SUSPENDIDO
2018	P_173	GAS NATURAL - SUSPENDIDO
2019	P_173	GAS NATURAL - SUSPENDIDO
2022	p_173	GAS NATURAL - SUSPENDIDO
2023	P_044	GAS NATURAL - SUSPENDIDO
2018	P_180	INTERNET - SUSPENDIDO
2019	P_180	INTERNET - SUSPENDIDO
2022	p_180	INTERNET - SUSPENDIDO
2023	P_051	INTERNET - SUSPENDIDO
2018	P_161	ENERGiA - DESCONECTADO
2019	P_161	ENERGiA - DESCONECTADO
2022	p_161	ENERGiA - DESCONECTADO
2023	P_032	ENERGiA - DESCONECTADO
2018	P_170	TELEFONO - DESCONECTADO
2019	P_170	TELEFONO - DESCONECTADO
2022	p_170	TELEFONO - DESCONECTADO
2023	P_041	TELEFONO - DESCONECTADO
2018	P_181	INTERNET - DESCONECTADO
2019	P_181	INTERNET - DESCONECTADO
2022	p_181	INTERNET - DESCONECTADO
2023	P_052	INTERNET - DESCONECTADO
2018	P_267	Usted cree que los hombres son mejores lideres politicos que las mujeres?
2019	P_267	Usted cree que los hombres son mejores lideres politicos que las mujeres?
2022	p_267	Usted cree que los hombres son mejores lideres politicos que las mujeres?
2023	P_144	Usted cree que los hombres son mejores lideres politicos que las mujeres?
2018	P_268	Obtener un Titulo Universitario es mas importante para un hombre que para una mujer?
2019	P_268	Obtener un Titulo Universitario es mas importante para un hombre que para una mujer?
2022	p_268	Obtener un Titulo Universitario es mas importante para un hombre que para una mujer?
2023	P_145	Obtener un Titulo Universitario es mas importante para un hombre que para una mujer?
2018	P_271	Usted considera que existe discriminacion contra la mujer?
2019	P_271	Usted considera que existe discriminacion contra la mujer?
2022	p_271	Usted considera que existe discriminacion contra la mujer?
2023	P_147	Usted considera que existe discriminacion contra la mujer?
2018	P_273	¿Usted considera que el Señor Alcalde es el que crea las leyes del municipio?
2019	P_273	¿Usted considera que el Señor Alcalde es el que crea las leyes del municipio?
2022	p_273	¿Usted considera que el Señor Alcalde es el que crea las leyes del municipio?
2023	P_149	¿Usted considera que el Señor Alcalde es el que crea las leyes del municipio?
2018	P_274	Usted considera que los juzgados son los encargados de elaborar las leyes?
2019	P_274	Usted considera que los juzgados son los encargados de elaborar las leyes?
2022	p_274	Usted considera que los juzgados son los encargados de elaborar las leyes?
2023	P_150	Usted considera que los juzgados son los encargados de elaborar las leyes?
2018	P_275	¿Usted considera que el Concejo Municipal, es el encargado de elegir Personero y Contralor municipales y de posesionarlo
2019	P_275	¿Usted considera que el Concejo Municipal, es el encargado de elegir Personero y Contralor municipales y de posesionarlo
2022	p_275	¿Usted considera que el Concejo Municipal, es el encargado de elegir Personero y Contralor municipales y de posesionarlo
2023	P_151	¿Usted considera que el Concejo Municipal, es el encargado de elegir Personero y Contralor municipales y de posesionarlo
2018	P_312	Califique Usted en una escala desde 1 hasta 5, la situación en su barrio o vereda, sobre: El estado de la contaminación
2019	P_312	Califique Usted en una escala desde 1 hasta 5, la situación en su barrio o vereda, sobre: El estado de la contaminación
2022	p_312	Califique Usted en una escala desde 1 hasta 5, la situación en su barrio o vereda, sobre: El estado de la contaminación
2023	P_175	Califique Usted en una escala desde 1 hasta 5, la situación en su barrio o vereda, sobre: El estado de la contaminación
2018	P_315	Califique Usted en una escala desde 1 hasta 5, la situación en su barrio o vereda, sobre: Las basuras y los escombros en
2019	P_315	Califique Usted en una escala desde 1 hasta 5, la situación en su barrio o vereda, sobre: Las basuras y los escombros en
2022	p_315	Califique Usted en una escala desde 1 hasta 5, la situación en su barrio o vereda, sobre: Las basuras y los escombros en
2023	P_177	Califique Usted en una escala desde 1 hasta 5, la situación en su barrio o vereda, sobre: Las basuras y los escombros en
2018	P_316	Califique Usted en una escala desde 1 hasta 5, la situación en su barrio o vereda, sobre: La contaminación visual
2019	P_316	Califique Usted en una escala desde 1 hasta 5, la situación en su barrio o vereda, sobre: La contaminación visual
2022	p_316	Califique Usted en una escala desde 1 hasta 5, la situación en su barrio o vereda, sobre: La contaminación visual
2023	P_178	Califique Usted en una escala desde 1 hasta 5, la situación en su barrio o vereda, sobre: La contaminación visual
2018	P_317	Califique Usted en una escala desde 1 hasta 5, la situación en su barrio o vereda, sobre: Arborización
2019	P_317	Califique Usted en una escala desde 1 hasta 5, la situación en su barrio o vereda, sobre: Arborización
2022	p_317	Califique Usted en una escala desde 1 hasta 5, la situación en su barrio o vereda, sobre: Arborización
2023	P_179	Califique Usted en una escala desde 1 hasta 5, la situación en su barrio o vereda, sobre: Arborización
2018	P_318	Califique Usted en una escala desde 1 hasta 5, la situación en su barrio o vereda, sobre: La pavimentación y señalizació
2019	P_318	Califique Usted en una escala desde 1 hasta 5, la situación en su barrio o vereda, sobre: La pavimentación y señalizació
2022	p_318	Califique Usted en una escala desde 1 hasta 5, la situación en su barrio o vereda, sobre: La pavimentación y señalizació
2023	P_180	Califique Usted en una escala desde 1 hasta 5, la situación en su barrio o vereda, sobre: La pavimentación y señalizació
2018	P_319	Califique Usted en una escala desde 1 hasta 5, la situación en su barrio o vereda, sobre: El estado de los andenes y de 
2019	P_319	Califique Usted en una escala desde 1 hasta 5, la situación en su barrio o vereda, sobre: El estado de los andenes y de 
2022	p_319	Califique Usted en una escala desde 1 hasta 5, la situación en su barrio o vereda, sobre: El estado de los andenes y de 
2023	P_181	Califique Usted en una escala desde 1 hasta 5, la situación en su barrio o vereda, sobre: El estado de los andenes y de 
2018	P_320	Califique Usted en una escala desde 1 hasta 5, la situación en su barrio o vereda, sobre: El cumplimiento de las normas 
2019	P_320	Califique Usted en una escala desde 1 hasta 5, la situación en su barrio o vereda, sobre: El cumplimiento de las normas 
2022	p_320	Califique Usted en una escala desde 1 hasta 5, la situación en su barrio o vereda, sobre: El cumplimiento de las normas 
2023	P_182	Califique Usted en una escala desde 1 hasta 5, la situación en su barrio o vereda, sobre: El cumplimiento de las normas 
2018	P_66	Tipo de afiliacion al sistema de seguridad social en salud
2019	P_66	Tipo de afiliacion al sistema de seguridad social en salud
2022	p_066	Tipo de afiliacion al sistema de seguridad social en salud
2023	P_242	Tipo de afiliacion al sistema de seguridad social en salud
2018	P_272	¿Cómo calificaría usted, en una escala desde 1 hasta 5, su nivel de satisfacción con su municipio?
2019	P_272	¿Cómo calificaría usted, en una escala desde 1 hasta 5, su nivel de satisfacción con su municipio?
2022	p_272	¿Cómo calificaría usted, en una escala desde 1 hasta 5, su nivel de satisfacción con su municipio?
2023	P_148	¿Cómo calificaría usted, en una escala desde 1 hasta 5, su nivel de satisfacción con su municipio?
2018	P_280	¿Cómo calificaría en una escala desde 1 hasta 5, su grado de confianza en las instituciones del gobierno?
2019	P_280	¿Cómo calificaría en una escala desde 1 hasta 5, su grado de confianza en las instituciones del gobierno?
2022	p_280	¿Cómo calificaría en una escala desde 1 hasta 5, su grado de confianza en las instituciones del gobierno?
2023	P_152	¿Cómo calificaría en una escala desde 1 hasta 5, su grado de confianza en las instituciones del gobierno?
2018	P_291	En los últimos 30 días, ¿Usted se preocupó alguna vez de que en su hogar se acabaran los alimentos debido a falta de din
2019	P_291	En los últimos 30 días, ¿Usted se preocupó alguna vez de que en su hogar se acabaran los alimentos debido a falta de din
2022	p_291	En los últimos 30 días, ¿Usted se preocupó alguna vez de que en su hogar se acabaran los alimentos debido a falta de din
2023	P_158	En los últimos 30 días, ¿Usted se preocupó alguna vez de que en su hogar se acabaran los alimentos debido a falta de din
2018	P_293	En los últimos 30 días, ¿alguna vez usted o algún adulto de su hogar no pudo variar la alimentación por falta de dinero?
2019	P_293	En los últimos 30 días, ¿alguna vez usted o algún adulto de su hogar no pudo variar la alimentación por falta de dinero?
2022	p_293	En los últimos 30 días, ¿alguna vez usted o algún adulto de su hogar no pudo variar la alimentación por falta de dinero?
2023	P_160	En los últimos 30 días, ¿alguna vez usted o algún adulto de su hogar no pudo variar la alimentación por falta de dinero?
2018	P_294	En los últimos 30 días, ¿Alguna vez usted o algun adulto de su hogar comió menos de lo que esta acostumbrado por falta d
2019	P_294	En los últimos 30 días, ¿Alguna vez usted o algun adulto de su hogar comió menos de lo que esta acostumbrado por falta d
2022	p_294	En los últimos 30 días, ¿Alguna vez usted o algun adulto de su hogar comió menos de lo que esta acostumbrado por falta d
2023	P_161	En los últimos 30 días, ¿Alguna vez usted o algun adulto de su hogar comió menos de lo que esta acostumbrado por falta d
2018	P_295	En los últimos 30 días, ¿Alguna vez en su hogar se quedaron sin alimentos por falta de dinero?
2019	P_295	En los últimos 30 días, ¿Alguna vez en su hogar se quedaron sin alimentos por falta de dinero?
2022	p_295	En los últimos 30 días, ¿Alguna vez en su hogar se quedaron sin alimentos por falta de dinero?
2023	P_162	En los últimos 30 días, ¿Alguna vez en su hogar se quedaron sin alimentos por falta de dinero?
2018	P_296	En los últimos 30 días, ¿Alguna vez usted o algún adulto de su hogar dejo de desayunar, almorzar o comer por falta de di
2019	P_296	En los últimos 30 días, ¿Alguna vez usted o algún adulto de su hogar dejo de desayunar, almorzar o comer por falta de di
2022	p_296	En los últimos 30 días, ¿Alguna vez usted o algún adulto de su hogar dejo de desayunar, almorzar o comer por falta de di
2023	P_164	En los últimos 30 días, ¿Alguna vez usted o algún adulto de su hogar dejo de desayunar, almorzar o comer por falta de di
2018	P_297	En los últimos 30 días, ¿alguna vez usted o algun adulto de su hogar sintió o se quejó de hambre y no comió por falta de
2019	P_297	En los últimos 30 días, ¿alguna vez usted o algun adulto de su hogar sintió o se quejó de hambre y no comió por falta de
2022	p_297	En los últimos 30 días, ¿alguna vez usted o algun adulto de su hogar sintió o se quejó de hambre y no comió por falta de
2023	P_165	En los últimos 30 días, ¿alguna vez usted o algun adulto de su hogar sintió o se quejó de hambre y no comió por falta de
2018	P_298	En los últimos 30 días, ¿Alguna vez usted o algún adulto de su hogar solo comió una sola vez al día o dejó de comer en t
2019	P_298	En los últimos 30 días, ¿Alguna vez usted o algún adulto de su hogar solo comió una sola vez al día o dejó de comer en t
2022	p_298	En los últimos 30 días, ¿Alguna vez usted o algún adulto de su hogar solo comió una sola vez al día o dejó de comer en t
2023	P_166	En los últimos 30 días, ¿Alguna vez usted o algún adulto de su hogar solo comió una sola vez al día o dejó de comer en t
2018	P_299	En los últimos 30 días, ¿Alguna vez, algún adulto de su hogar se acostó con hambre o porque no alcanzó el dinero para lo
2019	P_299	En los últimos 30 días, ¿Alguna vez, algún adulto de su hogar se acostó con hambre o porque no alcanzó el dinero para lo
2022	p_299	En los últimos 30 días, ¿Alguna vez, algún adulto de su hogar se acostó con hambre o porque no alcanzó el dinero para lo
2023	P_167	En los últimos 30 días, ¿Alguna vez, algún adulto de su hogar se acostó con hambre o porque no alcanzó el dinero para lo
2018	P_302	En los últimos 30 días, ¿Alguna vez usted tuvo que disminuir la cantidad servida en las comidas de algún niño o joven de
2019	P_302	En los últimos 30 días, ¿Alguna vez usted tuvo que disminuir la cantidad servida en las comidas de algún niño o joven de
2022	p_302	En los últimos 30 días, ¿Alguna vez usted tuvo que disminuir la cantidad servida en las comidas de algún niño o joven de
2023	P_170	En los últimos 30 días, ¿Alguna vez usted tuvo que disminuir la cantidad servida en las comidas de algún niño o joven de
2018	P_303	En los últimos 30 días, ¿Alguna vez, algun niño o joven de su hogar se quejó de hambre pero no se pudo comprar más alime
2019	P_303	En los últimos 30 días, ¿Alguna vez, algun niño o joven de su hogar se quejó de hambre pero no se pudo comprar más alime
2022	p_303	En los últimos 30 días, ¿Alguna vez, algun niño o joven de su hogar se quejó de hambre pero no se pudo comprar más alime
2023	P_171	En los últimos 30 días, ¿Alguna vez, algun niño o joven de su hogar se quejó de hambre pero no se pudo comprar más alime
2018	P_304	En los últimos 30 días, ¿ alguna vez algun niño o joven de su hogar se acostó con hambre porque no alcanzó el dinero par
2019	P_304	En los últimos 30 días, ¿ alguna vez algun niño o joven de su hogar se acostó con hambre porque no alcanzó el dinero par
2022	p_304	En los últimos 30 días, ¿ alguna vez algun niño o joven de su hogar se acostó con hambre porque no alcanzó el dinero par
2023	P_172	En los últimos 30 días, ¿ alguna vez algun niño o joven de su hogar se acostó con hambre porque no alcanzó el dinero par
2018	P_305	En los últimos 30 días, ¿Alguna vez, algun niño o joven de su hogar sólo comió una vez al día o dejó de comer todo un dí
2019	P_305	En los últimos 30 días, ¿Alguna vez, algun niño o joven de su hogar sólo comió una vez al día o dejó de comer todo un dí
2022	p_305	En los últimos 30 días, ¿Alguna vez, algun niño o joven de su hogar sólo comió una vez al día o dejó de comer todo un dí
2023	P_173	En los últimos 30 días, ¿Alguna vez, algun niño o joven de su hogar sólo comió una vez al día o dejó de comer todo un dí
2018	P_12	Incluyéndose usted, ¿Cuantas personas componen este hogar?
2019	P_12	Incluyéndose usted, ¿Cuantas personas componen este hogar?
2022	p_012	Incluyéndose usted, ¿Cuantas personas componen este hogar?
2023	P_058	Incluyéndose usted, ¿Cuantas personas componen este hogar?
2018	P_188	¿En cuántos, de los cuartos que son exclusivos para dormir, duermen las personas de este hogar?
2019	P_188	¿En cuántos, de los cuartos que son exclusivos para dormir, duermen las personas de este hogar?
2022	p_188	¿En cuántos, de los cuartos que son exclusivos para dormir, duermen las personas de este hogar?
2023	P_064	¿En cuántos, de los cuartos que son exclusivos para dormir, duermen las personas de este hogar?
2018	P_189	Nevera o enfriador
2019	P_189	Nevera o enfriador
2022	p_189	Nevera o enfriador
2023	P_071	Nevera o enfriador
2018	P_190	Lavadora de ropa
2019	P_190	Lavadora de ropa
2022	p_190	Lavadora de ropa
2023	P_072	Lavadora de ropa
2018	P_191	TV a blanco y negro
2019	P_191	TV a blanco y negro
2022	p_191	TV a blanco y negro
2023	P_073	TV a blanco y negro
2018	P_192	TV a color
2019	P_192	TV a color
2022	p_192	TV a color
2023	P_074	TV a color
2018	P_193	Calentador de agua o ducha eléctrica
2019	P_193	Calentador de agua o ducha eléctrica
2022	p_193	Calentador de agua o ducha eléctrica
2023	P_075	Calentador de agua o ducha eléctrica
2018	P_194	Calentador de agua a gas
2019	P_194	Calentador de agua a gas
2022	p_194	Calentador de agua a gas
2023	P_076	Calentador de agua a gas
2018	P_195	Estufa eléctrica
2019	P_195	Estufa eléctrica
2022	p_195	Estufa eléctrica
2023	P_077	Estufa eléctrica
2018	P_196	Estufa a gas
2019	P_196	Estufa a gas
2022	p_196	Estufa a gas
2023	P_078	Estufa a gas
2018	P_197	Estufa mixta
2019	P_197	Estufa mixta
2022	p_197	Estufa mixta
2023	P_079	Estufa mixta
2018	P_198	Parrilla a gas
2019	P_198	Parrilla a gas
2022	p_198	Parrilla a gas
2023	P_080	Parrilla a gas
2018	P_199	Parrilla eléctrica
2019	P_199	Parrilla eléctrica
2022	p_199	Parrilla eléctrica
2023	P_081	Parrilla eléctrica
2018	P_200	Horno microondas
2019	P_200	Horno microondas
2022	p_200	Horno microondas
2023	P_082	Horno microondas
2018	P_201	Horno eléctrico
2019	P_201	Horno eléctrico
2022	p_201	Horno eléctrico
2023	P_083	Horno eléctrico
2018	P_202	Horno a gas
2019	P_202	Horno a gas
2022	p_202	Horno a gas
2023	P_084	Horno a gas
2018	P_203	Equipo de sonido
2019	P_203	Equipo de sonido
2022	p_203	Equipo de sonido
2023	P_085	Equipo de sonido
2018	P_204	DVD
2019	P_204	DVD
2022	p_204	DVD
2023	P_086	DVD
2018	P_205	Computador de escritorio o portatil para uso del hogar, tableta
2019	P_205	Computador de escritorio o portatil para uso del hogar, tableta
2022	p_205	Computador de escritorio o portatil para uso del hogar, tableta
2023	P_087	Computador de escritorio o portatil para uso del hogar, tableta
2018	P_206	Servicio de TV por suscripción-TV por cable sátelital
2019	P_206	Servicio de TV por suscripción-TV por cable sátelital
2022	p_206	Servicio de TV por suscripción-TV por cable sátelital
2023	P_088	Servicio de TV por suscripción-TV por cable sátelital
2018	P_207	Celular
2019	P_207	Celular
2022	p_207	Celular
2023	P_089	Celular
2018	P_208	Aspiradora y/o Brilladora
2019	P_208	Aspiradora y/o Brilladora
2022	p_208	Aspiradora y/o Brilladora
2023	P_090	Aspiradora y/o Brilladora
2018	P_209	Aire acondicionado
2019	P_209	Aire acondicionado
2022	p_209	Aire acondicionado
2023	P_091	Aire acondicionado
2018	P_210	Consolas de video juegos o de juegos electrónicos
2019	P_210	Consolas de video juegos o de juegos electrónicos
2022	p_210	Consolas de video juegos o de juegos electrónicos
2023	P_092	Consolas de video juegos o de juegos electrónicos
2018	P_212	¿Cuántos vehículos particulares, en funcionamiento, tiene este hogar? (no incluye vehículo de servicio público o utiliza
2019	P_212	¿Cuántos vehículos particulares, en funcionamiento, tiene este hogar? (no incluye vehículo de servicio público o utiliza
2022	p_212	¿Cuántos vehículos particulares, en funcionamiento, tiene este hogar? (no incluye vehículo de servicio público o utiliza
2023	P_065	¿Cuántos vehículos particulares, en funcionamiento, tiene este hogar? (no incluye vehículo de servicio público o utiliza
2018	P_213	Cuántas Motos, motonetas
2019	P_213	Cuántas Motos, motonetas
2022	p_213	Cuántas Motos, motonetas
2023	P_066	Cuántas Motos, motonetas
2018	P_214	Cuántas Bicicletas
2019	P_214	Cuántas Bicicletas
2022	p_214	Cuántas Bicicletas
2023	P_067	Cuántas Bicicletas
2018	P_10	Estrato de la Vivienda
2019	P_10	Estrato de la Vivienda
2022	p_010	Estrato de la Vivienda
2023	P_014	Estrato de la Vivienda
2018	P_6	COMUNA O CORREGIMIENTO
2019	P_6	COMUNA O CORREGIMIENTO
2022	p_006	COMUNA O CORREGIMIENTO
2023	P_007	COMUNA O CORREGIMIENTO"""

df_map = pd.read_csv(io.StringIO(map_data), sep='\t')

# 2. Definir los nombres limpios personalizados de las dimensiones maestras
nombres_personalizados = {
    "COMUNA O CORREGIMIENTO": "COMUNA_STD",
    "Estrato de la Vivienda": "ESTRATO_STD",
    "ultimo NIVEL de estudio aprobado (titulo)": "EDUCACION_STD",
    "¿En cuántos, de los cuartos que son exclusivos para dormir, duermen las personas de este hogar?": "HABITACIONES_STD",
    "Incluyéndose usted, ¿Cuantas personas componen este hogar?": "PERSONAS_HOGAR_STD"
}

# Usar 2018 como la nomenclatura base (para las otras 81 variables)
nombres_2018 = df_map[df_map['Año'] == 2018].set_index('Pregunta')['Variable'].to_dict()

# Crear la columna destino: si tiene nombre personalizado lo usa, sino, usa el de 2018
df_map['Standard_Col'] = df_map['Pregunta'].apply(lambda x: nombres_personalizados.get(x, nombres_2018.get(x)))

# 3. Diccionario con la ruta de tus tablas en el Lakehouse
tablas_bronze = {
    2018: "Data_LakeHouse.dbo.ecv_bronze_2018",
    2019: "Data_LakeHouse.dbo.ecv_bronze_2019",
    2022: "Data_LakeHouse.dbo.ecv_bronze_2022",
    2023: "Data_LakeHouse.dbo.ecv_bronze_2023"
}

df_clean_list = []

# 4. Iterar, extraer de Fabric, mapear y limpiar por año
for año, tabla in tablas_bronze.items():
    # Leer mediante Spark SQL y bajar a Pandas (ejecutar en Notebook de Fabric)
    df_año = spark.sql(f"SELECT * FROM {tabla}").toPandas()
    
    if df_año.empty:
        continue
        
    # Forzar la creación de la columna YEAR (por seguridad)
    df_año['YEAR'] = año
    
    # Extraer el diccionario de renombramiento EXACTO para este año
    map_año_df = df_map[df_map['Año'] == año]
    dict_renames_año = dict(zip(map_año_df['Variable'], map_año_df['Standard_Col']))
    
    # Filtrar solo las variables que nos interesan (que estén en la tabla bronze original)
    cols_existentes = [col for col in dict_renames_año.keys() if col in df_año.columns]
    
    # Queremos YEAR + las variables filtradas
    cols_to_keep = ['YEAR'] + cols_existentes
    
    # Aplicar filtros y renombrar
    df_filtrado = df_año[cols_to_keep].copy()
    df_filtrado.rename(columns=dict_renames_año, inplace=True)
    
    df_clean_list.append(df_filtrado)

# 5. Consolidar en el DataFrame final
df_silver = pd.concat(df_clean_list, ignore_index=True)
print(f'🔥 Capa Silver creada con {df_silver.shape[0]} filas y exactamente {df_silver.shape[1]} columnas.')

# Opcional: Escribir la tabla Silver consolidada en el Lakehouse
spark.createDataFrame(df_silver).write.format("delta").mode("overwrite").saveAsTable("Data_LakeHouse.dbo.ecv_silver_consolidada")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df_silver

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

for i in df_silver.columns.tolist():
    print(f"--- Valores únicos para {i} ---")
    # Es necesario el print() dentro del bucle para que se muestre en consola
    print(df_silver[i].unique())
    print("\n") # Salto de línea para mayor legibilidad

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# # EDA

# CELL ********************

import pandas as pd
import numpy as np

# 1. Forzar tipado numérico en todas las columnas excepto en la Comuna
cols_to_numeric = df_silver.columns.drop(['COMUNA_STD'])
for col in cols_to_numeric:
    df_silver[col] = pd.to_numeric(df_silver[col], errors='coerce')

# 2. Estandarización de Comunas
comuna_map = {
    '1': '01 - Popular', 'POPULAR': '01 - Popular',
    '2': '02 - Santa Cruz', 'SANTA CRUZ': '02 - Santa Cruz',
    '3': '03 - Manrique', 'MANRIQUE': '03 - Manrique',
    '4': '04 - Aranjuez', 'ARANJUEZ': '04 - Aranjuez',
    '5': '05 - Castilla', 'CASTILLA': '05 - Castilla',
    '6': '06 - Doce de Octubre', 'DOCE DE OCTUBRE': '06 - Doce de Octubre',
    '7': '07 - Robledo', 'ROBLEDO': '07 - Robledo',
    '8': '08 - Villa Hermosa', 'VILLA HERMOSA': '08 - Villa Hermosa',
    '9': '09 - Buenos Aires', 'BUENOS AIRES': '09 - Buenos Aires',
    '10': '10 - La Candelaria', 'LA CANDELARIA': '10 - La Candelaria',
    '11': '11 - Laureles - Estadio', 'LAURELES-ESTADIO': '11 - Laureles - Estadio', 'LAURELES ESTADIO': '11 - Laureles - Estadio',
    '12': '12 - La America', 'LA AMERICA': '12 - La America',
    '13': '13 - San Javier', 'SAN JAVIER': '13 - San Javier',
    '14': '14 - El Poblado', 'EL POBLADO': '14 - El Poblado', 'POBLADO': '14 - El Poblado',
    '15': '15 - Guayabal', 'GUAYABAL': '15 - Guayabal',
    '16': '16 - Belen', 'BELEN': '16 - Belen',
    '50': '50 - San Sebastian de Palmitas', 'PALMITAS': '50 - San Sebastian de Palmitas',
    '60': '60 - San Cristobal', 'SAN CRISTOBAL': '60 - San Cristobal',
    '70': '70 - Altavista', 'ALTAVISTA': '70 - Altavista',
    '80': '80 - San Antonio de Prado', 'SAN ANTONIO DE PRADO': '80 - San Antonio de Prado',
    '90': '90 - Santa Elena', 'SANTA ELENA': '90 - Santa Elena'
}
df_silver['COMUNA_STD'] = df_silver['COMUNA_STD'].astype(str).str.strip().str.upper().map(comuna_map)

# 3. Corrección Estricta: Seguridad Alimentaria (Hambruna)
# Regla: 1=Sí (Aporta 1). 2=No (Aporta 0). 0 o -88=No aplica/No hay niños (Aporta 0). 
# Todo lo demás (8, -98 No sabe, -99 No responde) se volverá NaN al usar .map()
map_elcsa = {1: 1, 2: 0, 0: 0, -88: 0}
elcsa_cols = ['P_291', 'P_293', 'P_294', 'P_295', 'P_296', 'P_297', 'P_298', 'P_299', 'P_302', 'P_303', 'P_304', 'P_305']
for col in elcsa_cols:
    df_silver[col] = df_silver[col].map(map_elcsa)

# 4. Corrección Estricta: Programas de Recreación e Identidad de Género
map_binario = {1: 1, 2: 0, 0: 0, -88: 0}
for col in ['P_127', 'P_128', 'P_129', 'P_267', 'P_268', 'P_271']:
    df_silver[col] = df_silver[col].map(map_binario)

# 5. Corrección Estricta: Capital Físico (Conteos)
# -88 es 0 (No aplica). -98, -99 y 8 (cuando se usa como código de error) son nulos.
bienes_cols = ['P_189', 'P_190', 'P_191', 'P_192', 'P_193', 'P_194', 'P_195', 'P_196', 'P_197', 'P_198', 'P_199', 
               'P_200', 'P_201', 'P_202', 'P_203', 'P_204', 'P_205', 'P_206', 'P_207', 'P_208', 'P_209', 'P_210',
               'P_212', 'P_213', 'P_214']
for col in bienes_cols:
    df_silver[col] = df_silver[col].replace({-88: 0, -98: np.nan, -99: np.nan, 8: np.nan})

# 6. Variables de Escala y Satisfacción (Limitar de 1 a 5)
escala_cols = ['P_312', 'P_315', 'P_316', 'P_317', 'P_318', 'P_319', 'P_320', 'P_272', 'P_280', 
               'P_159', 'P_163', 'P_166', 'P_168', 'P_172', 'P_175', 'P_177', 'P_179']
for col in escala_cols:
    df_silver[col] = df_silver[col].apply(lambda x: np.nan if x < 1 or x > 5 else x)

# 7. Participación (Conocimiento cívico metodológico)
# Si responde -98 ("No sabe") la ley, es una respuesta incorrecta (Aporta 0). Si es -99 (No responde), es NaN.
df_silver['P_273'] = df_silver['P_273'].map({2: 1, 1: 0, -98: 0, -99: np.nan})
df_silver['P_274'] = df_silver['P_274'].map({2: 1, 1: 0, -98: 0, -99: np.nan})
df_silver['P_275'] = df_silver['P_275'].map({1: 1, 2: 0, -98: 0, -99: np.nan})

# 8. Hacinamiento y Variables Estructurales
df_silver['HABITACIONES_STD'] = df_silver['HABITACIONES_STD'].apply(lambda x: np.nan if x <= 0 else x)
df_silver['PERSONAS_HOGAR_STD'] = df_silver['PERSONAS_HOGAR_STD'].apply(lambda x: np.nan if x <= 0 else x)

print("Limpieza de anomalías finalizada. Tipado estricto y Nulls aplicados correctamente.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Capa Silver Enriquecida (Nivel Hogar)
df_silver_dims = pd.DataFrame()

df_silver_dims['YEAR'] = df_silver['YEAR']
df_silver_dims['COMUNA_STD'] = df_silver['COMUNA_STD']
df_silver_dims['ESTRATO_STD'] = df_silver['ESTRATO_STD']

df_silver_dims['EDUCACION_STD'] = df_silver['EDUCACION_STD']
df_silver_dims['ALFABETISMO'] = df_silver['P_35'].replace({2: 0, -98: np.nan, -99: np.nan}) 

# skipna=False propaga el NaN si falta información clave
df_silver_dims['DIM_RECREACION'] = df_silver[['P_127', 'P_128', 'P_129']].sum(axis=1, skipna=False)

df_silver_dims['IND_DISCRIMINACION'] = df_silver['P_271']
df_silver_dims['IND_BRECHA_PERCEPCION'] = df_silver[['P_267', 'P_268']].mean(axis=1)

df_silver_dims['DIM_PARTICIPACION'] = df_silver[['P_273', 'P_274', 'P_275']].sum(axis=1, skipna=False)

# Medio Ambiente y Movilidad (mean() excluye NaNs por defecto en el cálculo interno)
df_silver_dims['DIM_MEDIO_AMBIENTE'] = df_silver[['P_312', 'P_315', 'P_316', 'P_317']].mean(axis=1)
df_silver_dims['DIM_MOVILIDAD'] = df_silver[['P_318', 'P_319']].mean(axis=1)

df_silver_dims['SATISFACCION_MUNICIPIO'] = df_silver['P_272']
df_silver_dims['CONFIANZA_INSTITUCIONES'] = df_silver['P_280']
df_silver_dims['CUMPLIMIENTO_NORMAS'] = df_silver['P_320']

# Seguridad Alimentaria (Propaga NaN si el hogar no contestó)
df_silver_dims['DIM_SEGURIDAD_ALIMENTARIA'] = df_silver[elcsa_cols].sum(axis=1, skipna=False)

# Capital Físico
df_silver_dims['DIM_CAPITAL_FISICO'] = df_silver[bienes_cols].sum(axis=1, skipna=False)

# Acceso a Servicios Públicos (Índice de Carencia)
cobertura_cols = ['P_158', 'P_162', 'P_165', 'P_167', 'P_171', 'P_174', 'P_178']
penalidad_cobertura = (df_silver[cobertura_cols] == 2).sum(axis=1) * 3

calidad_cols = ['P_159', 'P_163', 'P_166', 'P_168', 'P_172', 'P_175', 'P_179']
penalidad_calidad = (df_silver[calidad_cols].isin([1, 2])).sum(axis=1) * 1

suspension_cols = ['P_160', 'P_164', 'P_169', 'P_173', 'P_180', 'P_161', 'P_170', 'P_181']
penalidad_suspension = (df_silver[suspension_cols] == 1).sum(axis=1) * 2

df_silver_dims['INDICE_CARENCIA_SERVICIOS'] = penalidad_cobertura + penalidad_calidad + penalidad_suspension

# Déficit Habitacional
df_silver_dims['IND_HACINAMIENTO'] = np.where((df_silver['PERSONAS_HOGAR_STD'] / df_silver['HABITACIONES_STD']) >= 3, 1, 0)
# Asegurar que si faltan datos de personas o cuartos, el hacinamiento sea nulo
df_silver_dims.loc[df_silver['HABITACIONES_STD'].isna() | df_silver['PERSONAS_HOGAR_STD'].isna(), 'IND_HACINAMIENTO'] = np.nan

print("Dimensiones procesadas a nivel de hogar.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 1. Agrupar por Año y Comuna calculando el Promedio de TODAS las variables numéricas
# Al dejar ESTRATO_STD, obtendremos el "Estrato Promedio" o índice socioeconómico de la comuna.
df_gold = df_silver_dims.groupby(['YEAR', 'COMUNA_STD']).mean().reset_index()

# 2. Redondear a 4 decimales para eficiencia y limpieza en el motor relacional
df_gold = df_gold.round(4)

# 3. Eliminar registros huérfanos sin comuna asignada (errores de recolección de origen)
df_gold = df_gold.dropna(subset=['COMUNA_STD'])

print(f'🏆 CAPA GOLD FINAL CREADA: {df_gold.shape[0]} registros promediados.')
display(df_gold.head())

# guardar directamente en tu Lakehouse en formato Delta:
spark.createDataFrame(df_gold).write.format("delta").mode("overwrite").saveAsTable("Data_LakeHouse.dbo.ecv_gold_promedios_comuna")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.functions import col, substring
from pyspark.sql.types import IntegerType

# 1. Cargar la tabla
df = spark.sql("SELECT * FROM Data_LakeHouse.dbo.ecv_gold_promedios_comuna")

# 2. Modificar la columna 'COMUNA_STD'
# Se extraen los 2 primeros caracteres y se aplica tipado estricto a Entero (int)
# Así "01 - Popular" primero pasa a "01" y luego se convierte al número 1.
df_modificado = df.withColumn(
    "COMUNA_STD", 
    substring(col("COMUNA_STD"), 1, 2).cast(IntegerType())
)

# Validar los datos visualmente
display(df_modificado)

# 3. Volver a guardar (sobrescribir) la tabla con el nuevo esquema (Texto -> Entero)
df_modificado.write \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("Data_LakeHouse.dbo.ecv_gold_promedios_comuna")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 1. Cargar la tabla
df = spark.sql("SELECT * FROM Data_LakeHouse.dbo.ecv_gold_promedios_comuna")

# Muestra una tabla con los valores únicos de la columna
df_unicos = df.select("COMUNA_STD").distinct()
display(df_unicos)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
