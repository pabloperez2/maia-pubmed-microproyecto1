# maia-pubmed-microproyecto1
1.	Problema que abordarán y su contexto
En la investigación médica, los artículos científicos generalmente se presentan en resúmenes estructurados, en los que cada oración cumple un rol específico dentro del razonamiento, como por ejemplo describir el contexto, los objetivos, los métodos, los resultados o las conclusiones del estudio. Sin embargo, en muchos repositorios y bases de datos, estos resúmenes están en un formato de texto plano, sin una segmentación explícita de sus componentes.

Esta falta de estructuración dificulta tareas como la revisión rápida del contenido, la búsqueda de información específica dentro de un resumen y el desarrollo de herramientas de apoyo a investigadores y profesionales de la salud. El problema se vuelve más relevante a medida que crece el volumen de publicaciones científicas, haciendo inviable un análisis manual eficiente de la información disponible.

En este contexto, la clasificación automática de oraciones dentro de resúmenes médicos surge como una solución viable para estructurar el contenido textual y facilitar su análisis. Usando técnicas de procesamiento de lenguaje natural y aprendizaje supervisado, es posible identificar el rol retórico de cada oración y organizar el abstract de forma automática. El modelo propuesto resulta útil porque permite transformar texto no estructurado en información más accesible, reutilizable y apta para sistemas de análisis posteriores, como motores de búsqueda especializados o sistemas de apoyo a la toma de decisiones clínicas.

2.	Pregunta de negocio y alcance del proyecto
La pregunta de negocio que guía este proyecto es:

¿Puede un sistema basado en procesamiento de lenguaje natural clasificar automáticamente las oraciones de resúmenes médicos en categorías retóricas que faciliten su análisis y comprensión?

El alcance del proyecto está centrado en el desarrollo de un prototipo funcional que reciba como entrada el texto de un abstract médico y produzca como salida la clasificación de cada oración en una de las categorías definidas (por ejemplo: Background, Objective, Methods, Results y Conclusions). El sistema se apoya en conjuntos de datos públicos previamente anotados, lo que nos permite tomar el problema como una tarea de clasificación supervisada.

El proyecto se basa exclusivamente en el dominio representado en los datos utilizados. Su enfoque es evaluar la viabilidad de una solución basada en modelos predictivos que permita clasificar automáticamente las oraciones de un abstract médico.
