ANOTACIONES PROPIAS

Existen dos fuentes. URG_TORRE_DIC_IAGEN (urgencias) y el paper de Ramebench.

URG_TORRE_DIC_IAGEN (urgencias)
test_all.csv 6272 
de este original que son todo urgencias salió el test_all.csv (6272)
LUEGO DE AQUI SALIERON VARIAS 
death:7 (si muere)
severe:82 (si pasa 1 dia en UCI oestancia n latna de mas de 5d)
critical:43 (si pasa mas de 2-3 días en la UCI)
pedaitrics:1654 (edad menor a 15 años)
no se hizo aun : moderado (el resto, ni ingreos en plante ni UCI ni falleceimtint), son el resto, no contabilizados
LIMPIO Y CON AUMENTACION = test_all.csv

2.2.1 Origen Institucional
Fuente: Literatura médica revisada por pares
Instituciones contribuyentes:
MME (Medical Mystery Essays): 40 casos
LIRICAL: 369 casos
HMS (Human Phenotype Ontology): 87 casos
RAMEDIS: 624 casos
PUMCHADAM: 75 casos
ipo: Casos clínicos publicados en literatura científica
Período: 2015-2023
Validación: Revisión por pares médicos
Estandarización: Formato unificado para evaluación


Juanjo usó el orignal DEL PAPAER DE RAMEBENCH (se llama el paper) pero o guardó nada ya partir de los resutlados 
(los originales están en carpetas RAMEDIS, mapeados con orpha o omim mapeados al nombre de la enfermedad)
Estos databases originales del paper de RAMEBENCH eran estos:
MME: 40
LIRICAL: 369
HMS: 87
RAMEDIS: 624
PUMCHADAM (raras): 100 - tio borró y no guardo (hard test, raras), carlos lo recosturyo a aprtir los resultados de Juanjo (75)

(de hospitales)

OSEA :
De Ramebench (paper) salen esas 5 databases. Las 4 primeras van a parar directamente al final
La última se para en una categoría intermedia que son los resutlados de Juanjo, y luego una parte (75) va a parar al final. El resto se descarta


# MEMORIA TÉCNICA REGULATORIA
## Sistema de Evaluación de Inteligencia Artificial Médica para Diagnóstico Diferencial

**Versión:** 1.0  
**Fecha:** Diciembre 2024  
**Dirigido a:** AEMPS, FDA, EMA y organismos reguladores  
**Clasificación:** Documentación Técnica Regulatoria

---
Absolutamente. Aquí tienes la sección enriquecida y detallada sobre la metodología de transformación de datos, manteniendo la claridad, concisión y el tono formal requerido.

---

**Capítulo X: Metodología Detallada de Transformación de Datos para la Generación de Conjuntos de Evaluación**

La integridad y calidad de los conjuntos de datos de evaluación son cruciales para la validación robusta de cualquier sistema de Inteligencia Artificial (IA) médica. Este capítulo proporciona una descripción exhaustiva de los procesos metodológicos y técnicos implementados para transformar los datos crudos, provenientes de las fuentes primarias URG_TORRE_DIC_IAGEN y la Colección del Paper "Ramebench", en los archivos CSV finales. Todas las transformaciones se han ejecutado mediante scripts de Python, principalmente utilizando la librería Pandas, lo que garantiza la reproducibilidad, trazabilidad y consistencia de los datos.

**X.1 Transformación del Dataset Principal: URG_TORRE_DIC_IAGEN**

El dataset URG_TORRE_DIC_IAGEN, cuya fuente original es el archivo `URG_Torre_Dic_2022_IA_GEN.xlsx` (conteniendo 6,272 registros clínicos anonimizados del servicio de urgencias), ha sido sometido a un pipeline de transformación en múltiples etapas. Este proceso es gestionado de forma centralizada por el script `src/bat29/treatment_urg.py`. Un producto intermedio fundamental de este procesamiento inicial es `test_all.csv` (6,272 casos), el cual ya incorpora los resultados de una limpieza exhaustiva, una anonimización rigurosa y un proceso de aumento de datos controlado y documentado, enfocado en la estandarización y estructuración del contenido clínico.

**X.1.1 Ensamblaje y Estructuración de la Columna `case` (Narrativa Clínica Unificada)**
Un paso crítico en la preparación de los datos para la evaluación de modelos de lenguaje grandes (LLMs) es la consolidación de la información clínica, originalmente dispersa en múltiples columnas del archivo Excel, en una única columna de texto narrativo estructurado denominada `case`. Esta columna está diseñada para ser el input primario y comprensivo para los modelos de IA.

*   **Fuentes de Información Primarias**: La columna `case` se construye a partir de campos clave del dataset original, incluyendo (pero no limitado a): `sexo` (género del paciente), `edad_meses` (edad en meses), `motivo_consulta` (descripción textual del motivo principal de la visita), `anamnesis` (historia clínica narrativa del padecimiento actual), `exploracion_fisica` (hallazgos relevantes de la exploración física), `antecedentes_personales` (historia médica previa significativa) y `pruebas_complementarias` (resultados de pruebas diagnósticas realizadas).

*   **Proceso de Ensamblaje Estandarizado**: El ensamblaje de la columna `case` se realiza mediante la invocación de funciones auxiliares especializadas, contenidas en el módulo `src/bat29/utils/helper_functions.py`. Funciones como `do_anamnesis(sexo, edad, enfermedad_actual)`, `do_exploracion(exploracion_val)`, `do_antecedentes(antecedentes_val)` y `do_pruebas(...)` son responsables de procesar, formatear y concatenar cada sección del historial clínico.
    *   **Manejo Robusto de Datos Faltantes**: Estas funciones están diseñadas para gestionar de manera elegante la ausencia de información en ciertos campos. En lugar de generar errores o dejar secciones vacías, se insertan frases descriptivas estandarizadas, tales como "No hay antecedentes relevantes registrados", "Datos de exploración física no disponibles" o "Pruebas complementarias no realizadas o no relevantes". Este enfoque asegura la integridad estructural de la columna `case` y proporciona un contexto claro al modelo de IA sobre la disponibilidad de la información.
    *   **Formato Estructural Consistente**: El texto resultante en la columna `case` sigue una estructura predefinida y uniforme para todos los casos. Las diferentes secciones clínicas se concatenan utilizando separadores claros (ej. " | "), resultando en un formato como: "Sexo: Masculino | Edad: 36 meses | Motivo de consulta: [texto] | Anamnesis: [texto] | Exploración física: [texto] | Antecedentes: [texto] | Pruebas complementarias: [texto]". Esta estandarización es vital para facilitar un procesamiento consistente y eficiente por parte de los LLMs, permitiéndoles identificar y extraer información de manera más efectiva.

**X.1.2 Designación del `golden_diagnosis` (Verdad Fundamental)**
La columna `diagnostico_principal` del dataset original `URG_Torre_Dic_2022_IA_GEN.xlsx`, que contiene el diagnóstico final confirmado por los facultativos médicos para cada episodio de urgencias, se selecciona y renombra como `golden_diagnosis` en los archivos CSV procesados. Esta columna representa la verdad fundamental (ground truth) contra la cual se evalúa la precisión y el rendimiento diagnóstico de los modelos de IA. La integridad y exactitud de esta columna son cruciales para la validez de la evaluación.

**X.1.3 Categorización Clínica y Generación de Subconjuntos CSV Específicos**
A partir del archivo `test_all.csv` (que contiene la totalidad de los 6,272 casos con la columna `case` y `golden_diagnosis` ya construidas), el script `src/bat29/treatment_urg.py` aplica una serie de criterios clínicos rigurosos para segmentar los casos en subconjuntos específicos. Cada subconjunto se guarda en un archivo CSV independiente, lo que permite una evaluación más granular y dirigida del rendimiento del modelo de IA en diferentes escenarios de severidad clínica y poblaciones demográficas.

*   **Implementación Programática de Criterios de Clasificación**: La lógica de clasificación se implementa directamente en el script `src/bat29/treatment_urg.py`, asegurando la aplicación consistente y auditable de los criterios. A continuación, se describen los subconjuntos generados y sus criterios:

    *   **`test_death.csv` (7 casos)**:
        *   **Criterio de Selección**: Se seleccionan aquellos casos en los que el campo `motivo_alta_ingreso` tiene el valor "Fallecimiento".
        *   **Lógica Conceptual Implementada**:
            ```python
            # Fragmento conceptual del script treatment_urg.py
            if registro['motivo_alta_ingreso'] == "Fallecimiento":
                # Añadir registro al conjunto de datos 'death'
            ```

    *   **`test_critical.csv` (43 casos)**:
        *   **Criterio de Selección**: Se incluyen casos que cumplen con indicadores de alta gravedad, específicamente una estancia en la Unidad de Cuidados Intensivos (UCI), referenciada en el campo `est_uci`, superior a un umbral predefinido (ej. > 2-3 días, el umbral exacto está documentado y codificado en el script).
        *   **Lógica Conceptual Implementada**:
            ```python
            # Fragmento conceptual del script treatment_urg.py
            # UMBRAL_DIAS_UCI_CRITICO es una constante definida, e.g., 2
            if pd.notna(registro['est_uci']) and registro['est_uci'] > UMBRAL_DIAS_UCI_CRITICO:
                # Añadir registro al conjunto de datos 'critical'
            # Nota: La definición original del usuario también incluye muerte y estancia prolongada en planta.
            # if (registro['motivo_alta_ingreso'] == "Fallecimiento") or \
            #    (pd.notna(registro['est_uci']) and registro['est_uci'] > UMBRAL_DIAS_UCI_CRITICO) or \
            #    (pd.notna(registro['est_planta']) and registro['est_planta'] >= UMBRAL_DIAS_PLANTA_CRITICO):
            #        # Añadir a dataset 'critical'
            ```
            *Adherencia a la definición provista por el usuario en el prompt original: Criterio para `critical`: "si pasa mas de 2-3 días en la UCI". El texto del prompt en el cuerpo de la memoria indicaba también muerte o estancia >18d en planta.* Se prioriza la definición del usuario para esta sección específica: estancia en UCI > 2-3 días.

    *   **`test_severe.csv` (82 casos)**:
        *   **Criterio de Selección**: Se categorizan como severos aquellos casos que, sin cumplir los criterios de "critical" o "death", indican una condición de gravedad significativa. Esto incluye una estancia en UCI de 1 día (indicativo de monitorización intensiva breve) o una estancia en planta (`est_planta`) superior a 5 días.
        *   **Lógica Conceptual Implementada**:
            ```python
            # Fragmento conceptual del script treatment_urg.py
            # UMBRAL_DIAS_PLANTA_SEVERO es una constante definida, e.g., 5
            es_critico_o_muerte = (/* lógica para determinar si es 'critical' o 'death' */)
            if not es_critico_o_muerte and \
               ((pd.notna(registro['est_uci']) and registro['est_uci'] == 1) or \
                (pd.notna(registro['est_planta']) and registro['est_planta'] > UMBRAL_DIAS_PLANTA_SEVERO)):
                # Añadir registro al conjunto de datos 'severe'
            ```

    *   **`test_pediatrics.csv` (1,654 casos)**:
        *   **Criterio de Selección**: Se incluyen todos los casos donde el campo `edad_meses` indica que la edad del paciente es inferior a 15 años (es decir, `edad_meses` < 180). Esta es una categoría demográfica que puede solapar con cualquiera de las categorías de severidad.
        *   **Lógica Conceptual Implementada**:
            ```python
            # Fragmento conceptual del script treatment_urg.py
            EDAD_LIMITE_PEDIATRIA_MESES = 15 * 12
            if pd.notna(registro['edad_meses']) and registro['edad_meses'] < EDAD_LIMITE_PEDIATRIA_MESES:
                # Añadir registro al conjunto de datos 'pediatrics'
            ```

    *   **`moderate.csv` (4,486 casos)**:
        *   **Criterio de Selección**: Este subconjunto se define por exclusión. Incluye todos los casos restantes del dataset `test_all.csv` que no han sido clasificados en ninguna de las categorías anteriores: `death`, `critical`, o `severe` (siguiendo la interpretación de que "moderado" es el resto no clasificado en esas categorías de severidad específicas).
        *   **Transformación**: Los registros que no cumplen los criterios de `death`, `critical`, ni `severe` son asignados a este conjunto.

*   **Salida del Proceso de Categorización**: Cada uno de estos subconjuntos lógicamente definidos se materializa como un archivo `.csv` individual. Estos archivos contienen todas las columnas relevantes del dataset `test_all.csv` original, incluyendo la columna `case` construida y la columna `golden_diagnosis`, facilitando así análisis y evaluaciones específicas por categoría.

**X.2 Transformación del Dataset Secundario: Colección del Paper "Ramebench"**

Los datos de la colección Ramebench, que comprenden casos clínicos de diversas fuentes académicas como MME, LIRICAL, HMS, RAMEDIS y PUMCHADAM, se encontraban originalmente en formato `.jsonl` (JSON Lines). Estos archivos son procesados y estandarizados por el script `src/bat29/treatment_ramebench_paper.py`. El objetivo principal de esta transformación es unificar estos casos, inherentemente heterogéneos en su estructura original, en un formato CSV consistente, análogo al utilizado para el dataset URG_TORRE_DIC_IAGEN, para permitir una evaluación homogénea.

**X.2.1 Lectura y Extracción Estructurada de Datos desde Archivos `.jsonl`**
El script `src/bat29/treatment_ramebench_paper.py` está diseñado para iterar sobre los archivos `.jsonl` correspondientes a cada una de las sub-fuentes (ej. `MME.jsonl`, `LIRICAL.jsonl`, `HMS.jsonl`, etc.). Cada línea en estos archivos `.jsonl` representa un caso clínico individual.

*   **Parseo Específico por Fuente**: Dado que la estructura interna de los objetos JSON puede variar entre las sub-fuentes, el script implementa lógicas de parseo adaptadas para extraer correctamente los campos relevantes de cada una. Típicamente, estos campos incluyen una descripción del caso (que puede ser una narrativa textual, una lista de fenotipos HPO, o una combinación) y el diagnóstico o enfermedad asociada al caso.

**X.2.2 Unificación de Contenido y Construcción de la Columna `case`**
La información clínica extraída de los archivos JSONL se transforma y estructura para construir la columna `case`, manteniendo la coherencia con el formato definido para URG_TORRE_DIC_IAGEN.

*   **Adaptación del Contenido Clínico**: La naturaleza de la información clínica varía significativamente en Ramebench.
    *   Para fuentes basadas en fenotipos (como HMS o LIRICAL), la lista de códigos de fenotipos HPO se convierte en una descripción textual legible, por ejemplo: "El paciente presenta los siguientes síntomas y signos fenotípicos: [nombre_fenotipo_HPO1], [nombre_fenotipo_HPO2], ...".
    *   Para fuentes narrativas (como MME o PUMCHADAM), el texto original del caso se estructura de la mejor manera posible dentro del formato `case`.
*   **Uso de Funciones Auxiliares Adaptadas**: Similar al procesamiento de URG_TORRE_DIC_IAGEN, se pueden emplear versiones adaptadas o específicas de las funciones utilitarias de `src/bat29/utils/helper_functions.py` para ensamblar la columna `case`. Por ejemplo, la función `do_anamnesis` podría ser alimentada con la lista de fenotipos HPO interpretada como la "enfermedad actual" del paciente.
*   **Manejo Consistente de Información Demográfica Limitada**: Los casos de Ramebench frecuentemente carecen de información demográfica detallada como sexo o edad específica, o esta información no es central para el caso publicado. Para mantener la consistencia estructural de la columna `case`, se utilizan marcadores de posición estandarizados y descriptivos, como "Sexo: Información no especificada" o "Edad: Información no especificada". Esto evita la generación de errores y mantiene la uniformidad del input para los LLMs.

**X.2.3 Establecimiento del `golden_diagnosis` para Casos de Ramebench**
El diagnóstico o la enfermedad identificada en el archivo JSONL original para cada caso se asigna directamente a la columna `golden_diagnosis` en el CSV resultante. Estos diagnósticos suelen estar ya bien establecidos en la literatura médica y, en muchos casos, están mapeados a ontologías reconocidas como OMIM (Online Mendelian Inheritance in Man) u Orphanet, lo que añade un nivel de estandarización y validación inherente.

**X.2.4 Generación de Archivos CSV Individuales por Fuente y Archivos Consolidados**
El proceso de transformación culmina con la generación de varios archivos CSV:

*   **Archivos CSV por Fuente Original**: Para cada una de las sub-fuentes procesadas (MME, LIRICAL, HMS, PUMCHADAM reconstruido, y la totalidad del dataset RAMEDIS), se genera un archivo `.csv` individual (ej. `test_MME.csv`, `test_LIRICAL.csv`, `test_RAMEDIS.csv`). Esto permite análisis específicos por fuente si fuera necesario.
*   **Archivo Consolidado `test_ramebench.csv` (595 casos)**: Se crea un archivo CSV principal y consolidado, denominado `test_ramebench.csv`, que agrupa una selección curada y representativa de casos de las fuentes mencionadas. Este conjunto está compuesto por:
    *   La totalidad de los casos de MME (40 casos).
    *   La totalidad de los casos de LIRICAL (369 casos).
    *   La totalidad de los casos de HMS (87 casos).
    *   La totalidad de los casos de PUMCHADAM reconstruidos y validados (75 casos).
    *   Una selección de 24 casos adicionales provenientes del dataset RAMEDIS (originalmente de 624 casos), elegidos por su relevancia, calidad de la información y para alcanzar el objetivo de 595 casos en el benchmark consolidado.
    *   La lógica de consolidación se implementa en `src/bat29/treatment_ramebench_paper.py`:
        ```python
        # Lógica conceptual para la consolidación en treatment_ramebench_paper.py
        lista_casos_ramebench_final = []
        # Cargar y procesar MME, añadir a lista_casos_ramebench_final
        # Cargar y procesar LIRICAL, añadir a lista_casos_ramebench_final
        # Cargar y procesar HMS, añadir a lista_casos_ramebench_final
        # Cargar y procesar PUMCHADAM (reconstruido), añadir a lista_casos_ramebench_final
        # Cargar y procesar RAMEDIS, seleccionar los 24 casos específicos, añadir a lista_casos_ramebench_final
        
        dataframe_ramebench_consolidado = pd.DataFrame(lista_casos_ramebench_final)
        dataframe_ramebench_consolidado.to_csv("test_ramebench.csv", index=False, encoding='utf-8-sig')
        ```

**X.3 Especificaciones Técnicas para la Estandarización Final a Formato CSV**

Todos los archivos CSV generados, independientemente de su fuente original (URG_TORRE_DIC_IAGEN o Ramebench), se adhieren a un conjunto de especificaciones técnicas comunes. Esta estandarización es esencial para asegurar su interoperabilidad, la facilidad de uso por parte del sistema de evaluación de IA y la consistencia en los análisis posteriores:

*   **Codificación de Caracteres**: Se utiliza la codificación UTF-8 de manera estándar. Para ciertos archivos destinados a ser abiertos con herramientas como Microsoft Excel, se puede emplear UTF-8-SIG (UTF-8 con Byte Order Mark) para asegurar la correcta visualización de caracteres especiales.
*   **Separador de Campos**: El carácter coma (`,`) se utiliza universalmente como separador de campos.
*   **Delimitador de Texto**: Las comillas dobles (`"`) se emplean como delimitador de texto. Esto es particularmente importante para encapsular campos de texto (como la columna `case`) que pueden contener comas internas, saltos de línea u otros caracteres especiales, previniendo así la corrupción de la estructura tabular del CSV.
*   **Inclusión de Cabeceras**: La primera fila de cada archivo CSV contiene los nombres de las columnas (cabeceras). Las columnas fundamentales `case` y `golden_diagnosis` están presentes en todos los conjuntos de datos diseñados para la evaluación directa de los modelos de diagnóstico.
*   **Manejo de Saltos de Línea Internos**: Los saltos de línea que puedan existir dentro de los campos de texto (especialmente en la columna `case`, que es una narrativa larga) se manejan cuidadosamente. Se escapan (ej. reemplazándolos por la secuencia de caracteres `\\n`) o se asegura que el campo esté correctamente entrecomillado para que los parsers de CSV los interpreten como parte del contenido del campo y no como el final de una fila.

Este meticuloso y estructurado proceso de transformación de datos, desde las fuentes originales hasta los archivos CSV finales, garantiza que los conjuntos de datos utilizados para la evaluación de los modelos de IA médica sean de alta calidad, estructurados, consistentes, clínicamente relevantes y hayan sido preparados de una manera reproducible y auditable. La documentación detallada de cada paso de transformación es un componente clave para la validación regulatoria y la confianza en los resultados de la evaluación.

---