# Corpus de POIs

## Estado actual

El fichero [pois_bilbao_bizkaia.json](<d:/MASTER/MPL/NLPProyecto/data/pois_bilbao_bizkaia.json>) contiene el corpus completo que usa la aplicación en runtime.

- Total de POIs: `1109`
- POIs de Bilbao: `465`
- POIs del resto de Bizkaia: `644`
- Fuentes integradas:
  - `342` de `OpenStreetMap Overpass`
  - `74` de `Open Data Euskadi`
  - `693` de `Wikidata`

La aplicación no hace llamadas externas para completar el corpus al arrancar. Igual que antes cargaba un JSON pequeño, ahora carga directamente este JSON ampliado y reindexa solo si detecta que ha cambiado.

## Qué he hecho

1. He eliminado la dependencia de una base manual de 40 POIs como punto de partida conceptual.
2. He dejado [pois_bilbao_bizkaia.json](<d:/MASTER/MPL/NLPProyecto/data/pois_bilbao_bizkaia.json>) como fuente única de verdad para la app.
3. He añadido un generador reproducible en [scripts/expand_bilbao_corpus.py](<d:/MASTER/MPL/NLPProyecto/scripts/expand_bilbao_corpus.py>) para reconstruir el corpus.
4. He ampliado el corpus con dos fuentes estructuradas nuevas y masivas:
   - Open Data Euskadi / Open Data Bilbao
   - Wikidata
5. He mantenido la carga en runtime totalmente local, sin llamadas HTTP al arrancar.

## Cómo se ha construido

El proceso ya no parte de POIs escritos a mano. Parte de fuentes abiertas y normaliza todo al mismo esquema del proyecto.

### 1. Capa OSM

La parte de OpenStreetMap se reaprovecha desde la capa ya materializada en el JSON, que a su vez procede de Overpass. He dejado esa capa como base automática para no depender de la disponibilidad puntual del servicio Overpass cada vez que se regenere el corpus.

### 2. Open Data Euskadi

El script descarga el RDF oficial de lugares de interés turístico de Bilbao:

- `https://www.bilbao.eus/bilbaoopendata/turismo/lugares_interes_turistico.rdf`

Con esa fuente:

- se extraen nombre, dirección y tipo oficial
- se convierten las coordenadas UTM del RDF a WGS84
- se mapean los tipos al esquema del proyecto
- se eliminan duplicados contra OSM

Resultado en esta iteración: `74` POIs nuevos útiles tras deduplicado.

### 3. Wikidata

El script consulta Wikidata por lotes usando SPARQL para clases turísticas y patrimoniales relevantes de Bilbao y Bizkaia.

Ejemplos de clases consultadas:

- museos
- puentes
- parques y jardines
- monumentos
- esculturas y estatuas
- iglesias, ermitas y monasterios
- castillos y sitios arqueológicos
- patrimonio industrial
- faros, palacios, miradores y funiculares

Después del fetch:

- se parsean coordenadas geográficas
- se normalizan nombres y municipios
- se filtran elementos poco útiles para turismo
  - bibliotecas genéricas
  - vértices geodésicos
  - fosas comunes
  - elementos claramente funerarios o administrativos
- se deduplican contra OSM y Open Data

Resultado en esta iteración: `693` POIs añadidos desde Wikidata.

### 4. Deduplicado global

El deduplicado se hace con varias señales:

- nombre normalizado
- solapamiento de tokens significativos
- distancia geográfica
- categoría compatible

Así evitamos meter varias veces el mismo museo, puente o monumento aunque aparezca en más de una fuente.

### 5. Normalización final

Todos los registros acaban con el mismo esquema:

- `id`
- `name`
- `municipality`
- `category`
- `subcategory`
- `description`
- `coordinates`
- `address`
- `price` y `price_numeric`
- `schedule`
- `source`
- `url`
- `tags`
- `enriched_text`
- `visit_duration_minutes`
- `accessibility`

## Qué carga ahora la app

[app/poi_manager.py](<d:/MASTER/MPL/NLPProyecto/app/poi_manager.py>) carga directamente el JSON final, lo convierte a objetos `POI` y reindexa ChromaDB solo cuando cambia la firma del corpus.

No hay:

- ampliación dinámica en runtime
- llamadas HTTP al arrancar
- caché temporal del corpus descargado

## Cómo regenerarlo

Desde la raíz del proyecto:

```bash
python scripts/expand_bilbao_corpus.py
```

Si solo quieres ver el resumen sin escribir el fichero:

```bash
python scripts/expand_bilbao_corpus.py --dry-run
```

## Criterio de cantidad

Para este proyecto, una cantidad "buena" de POIs útiles y todavía razonablemente limpia está en torno a `600-800`.

He dejado el corpus en `1109` porque el objetivo aquí era maximizar cobertura y eliminar la dependencia de la base manual. Aun así, el script ya filtra bastante ruido y deja fuera varios tipos que empeoraban la calidad para turismo.
