# Corpus de POIs de Bilbao / Bizkaia

## Objetivo

Ampliar el corpus `pois_bilbao_bizkaia.json` para aumentar de forma clara la cobertura de **POIs del municipio de Bilbao** manteniendo el mismo esquema que ya consume la aplicación.

## Resultado

- Antes: `40` POIs totales, de los cuales `20` eran de Bilbao.
- Después: `394` POIs totales, de los cuales `374` son de Bilbao.
- Incremento en Bilbao: `+354` POIs nuevos.

La ampliación se hizo el **2026-04-17** con un flujo reproducible y no manual.

## Ficheros implicados

- `data/pois_bilbao_bizkaia.json`
- `scripts/expand_bilbao_corpus.py`

## Cómo lo he hecho

1. Partí del corpus existente y mantuve intactos los 40 registros originales.
2. Obtuve el límite administrativo real de Bilbao con **Nominatim** para no depender sólo de un rectángulo geográfico.
3. Consulté **OpenStreetMap** a través de **Overpass** para descargar candidatos dentro del bbox de Bilbao.
4. Filtré los elementos descargados para quedarme con POIs visitables o útiles para recuperación turística:
   - `tourism`: `museum`, `gallery`, `attraction`, `artwork`, `viewpoint`
   - `historic`: `monument`, `memorial`, `castle`, `archaeological_site`, `ruins`, `building`
   - `amenity`: `theatre`, `arts_centre`, `cinema`, `marketplace`, `place_of_worship`
   - `leisure`: `park`, `garden`
   - `man_made`: `bridge`
5. Eliminé ruido básico:
   - Alojamientos (`hotel`, `hostel`, `guest_house`, etc.)
   - Elementos privados
   - Elementos abandonados o fuera del polígono real de Bilbao
6. Hice deduplicado contra el corpus previo y entre nuevos candidatos:
   - Coincidencia exacta por nombre normalizado
   - Coincidencia aproximada por nombre + proximidad geográfica + categoría
7. Normalicé cada candidato al esquema del proyecto:
   - `category` y `subcategory`
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
8. Generé texto semántico adicional para mejorar la recuperación vectorial sin tocar la lógica del backend.

## Heurísticas que he aplicado

Como OpenStreetMap no siempre trae todos los campos que necesita el proyecto, añadí reglas sencillas y reproducibles:

- `schedule`:
  - POIs exteriores como puentes, parques, miradores, esculturas o memoriales: `00:00–23:59`
  - Museos: lunes cerrado y horario tipo `10:00–19:00`
  - Mercados: horario tipo `08:30–14:30`
  - Equipamientos culturales y religiosos: horario genérico compatible con itinerarios diurnos
- `price`:
  - Espacios exteriores y abiertos: `gratis`
  - Museos y espacios culturales con `fee=yes` o sin dato fiable: precio bajo estimado para no romper el planificador
- `visit_duration_minutes`:
  - Según tipo de POI: museo, parque, mercado, arte público, monumento, etc.
- `accessibility`:
  - Se respeta `wheelchair=*` cuando existe
  - Si no existe, se asigna un valor heurístico por categoría

Estas reglas están pensadas para que el corpus siga siendo utilizable por el planificador y el recuperador, no para sustituir una curación editorial completa.

## Script de regeneración

El proceso queda automatizado en:

```bash
python scripts/expand_bilbao_corpus.py
```

Modo sólo resumen:

```bash
python scripts/expand_bilbao_corpus.py --dry-run
```

El script:

- descarga el polígono de Bilbao desde Nominatim
- consulta Overpass con reintentos automáticos
- filtra y deduplica
- escribe el JSON final en `data/pois_bilbao_bizkaia.json`

## Validación realizada

Tras regenerar el fichero comprobé:

- que el JSON carga correctamente
- que hay `394` IDs únicos
- que todos los POIs tienen las claves esperadas
- que `schedule` contiene los 7 días en el formato que espera el proyecto

## Nota importante sobre el índice vectorial

El corpus ya está actualizado, pero si el sistema ya tenía una base ChromaDB previa, hay que **reindexar** para que los nuevos POIs entren en búsqueda semántica.

Opciones:

```bash
# Si la API está levantada
curl -X POST http://localhost:8000/api/admin/reindex
```

o bien borrar/regenerar la base de `db/chroma_db` si se quiere reconstruir desde cero en el siguiente arranque.

## Fuentes utilizadas

- OpenStreetMap Nominatim: https://nominatim.openstreetmap.org/
- OpenStreetMap Overpass API: https://overpass-api.de/
- OpenStreetMap: https://www.openstreetmap.org/
