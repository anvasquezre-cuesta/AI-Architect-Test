# Technology Comparison & Tradeoffs

Este documento compara las opciones de tecnología mencionadas en la prueba, sus tradeoffs, y justifica las decisiones tomadas en la solución.

---

## 1. OCR (Optical Character Recognition)

### Opciones

| Tecnología | Tipo | DPI Nativo | Velocidad | Accuracy | Bounding Box | Costo | 
|---|---|---|---|---|---|---|
| **Tesseract** | Open source (local) | 150-300 | Rápido (< 1s/página) | 85-95% | ✅ Sí, confiable | Gratis |
| **EasyOCR** | Open source (PyTorch) | Configurable | Lento (2-5s/página) | 90-98% | ✅ Sí | Gratis |
| **Cloud OCR** (Google Vision, AWS Textract) | API propietaria | N/A | Instantáneo | 98%+ | ✅ Sí, excelente | $$$$ por volumen |
| **PyMuPDF** (fitz) | Extracción (no OCR puro) | N/A | Muy rápido | Solo PDF digital | ❌ No | Gratis |

### Tradeoffs

**Tesseract:**
- ✅ Fast, deterministic, buen balance accuracy/speed
- ✅ Bounding boxes confiables para matching de nombres
- ❌ Necesita instalación binaria (no solo pip)
- ❌ Accuracy degrada con imágenes de baja calidad

**EasyOCR:**
- ✅ Mejor accuracy en general (deep learning)
- ✅ Maneja idiomas múltiples bien
- ❌ Mucho más lento (impactante a escala)
- ❌ Requiere GPU para performance razonable

**Cloud OCR:**
- ✅ Excelente accuracy y features (layout detection, tables)
- ✅ Cero mantenimiento
- ❌ Costo prohibitivo a 1000 PDFs/hora
- ❌ Latencia de red, compliance/privacy issues
- ❌ No funciona offline

**PyMuPDF (sin OCR):**
- ✅ Ultrarrápido para PDFs digitales
- ❌ No funciona con PDFs escaneados (el requisito de la prueba)

### Decisión: **Tesseract**

**Por qué:** El enunciado pide explícitamente "scanned PDFs", y Tesseract es el estándar industria para eso en soluciones locales. Balance óptimo de speed (< 1s), accuracy (85-95%), y costo a escala. Las bounding boxes que devuelve son confiables para el fuzzy matching posterior.

### Visualización: OCR — Speed vs Accuracy

```
Accuracy (%)
100 │                                  Cloud OCR
    │                                     ●
 95 │
    │                         EasyOCR
 90 │                            ●
    │
 85 │
    │                  Tesseract ✓
 80 │                     ●
    │
 75 │                 PyMuPDF (solo digital)
    │                     ●
    ├─────┬──────┬──────┬──────┬──────┬──────┤
    0.1s  1s     2s     3s     5s     10s   Speed/página
```

**Takeaway:** Tesseract es óptimo para esta prueba. EasyOCR si accuracy es crítica. Cloud si dinero sin límite.

---

## 2. NER (Named Entity Recognition)

### Opciones

| Tecnología | Approach | Accuracy | Latencia | Config | Costo | 
|---|---|---|---|---|---|
| **spaCy** | Trained statistical model (small) | 80-85% | ~50ms | Bajo | Gratis |
| **spaCy Large** | Trained statistical model (large) | 88-92% | ~150ms | Bajo | Gratis |
| **BERT/RoBERTa** | Transformer fine-tuned | 90-95% | ~500ms | Medio | Gratis |
| **LLM Zero-shot** (GPT, Claude, Ollama) | Generative prompt | 95%+ | 500ms-3s | Alto | $$ o compute local |
| **Custom trained NER** | Domain-specific model | 95%+ | ~100ms | Alto | $$$$ (training) |

### Tradeoffs

**spaCy (small):**
- ✅ Rápido, deterministic, bajo overhead
- ✅ Production-ready, ampliamente usado
- ❌ Accuracy moderada (pero OK para nombres comunes)
- ❌ Solo PERSON entities confiables

**spaCy (large):**
- ✅ Mejor accuracy que small
- ✅ Aún rápido (< 200ms)
- ❌ Más pesado (descarga, memoria)
- ❌ Overkill para tarea simple

**BERT/RoBERTa:**
- ✅ High accuracy
- ✅ Mejor que spaCy en edge cases
- ❌ Lento para volumen (500ms-1s per doc)
- ❌ Requiere modelo fine-tuned específico

**LLM Zero-shot:**
- ✅ Flexibility extrema ("dame nombres de personas y ejecutivos")
- ✅ Mejor en OCR noisy (entiende contexto)
- ❌ No deterministic (puede variar respuesta)
- ❌ Lento (500ms-3s depende del modelo)
- ❌ Costo por token o compute local requerido
- ❌ Overkill para tarea simple

**Custom trained NER:**
- ✅ Precision máxima para dominio específico
- ❌ Requiere dataset de entrenamiento (expensive)
- ❌ Maintenance burden

### Decisión: **spaCy (small)**

**Por qué:** 
- El enunciado da spaCy como ejemplo: *"Use spaCy's pre-trained NER model"*
- Para nombres de personas (la tarea), el modelo small de spaCy ya tiene 85%+ accuracy
- El OCR + fuzzy matching downstream ya tolera algunos errores
- Speed es crítico a escala (1000 PDFs/hora = ~280ms per doc; no podemos gastar 1-3s en NER)
- Zero-shot LLM es interesante pero lo dejarías para queries del RAG donde la flexibility vale más

### Visualización: NER — Speed vs Accuracy vs Cost

```
Accuracy (%)
100 │  Custom Trained
    │      ●
 95 │
    │      BERT/LLM
 90 │      ●
    │   spaCy Large
 85 │     ●
    │   spaCy Small ✓  Zero-shot LLM
 80 │   ●               ●
    │
    ├─────┬──────┬──────┬──────┬──────┬──────┤
  $0  $0 compute  $0.1  $1    $10   $100  Costo ($/1000 docs)

Speed (ms per doc):
spaCy Small: 50ms ✓
spaCy Large: 150ms
BERT: 500ms
LLM: 1000ms
```

**Takeaway:** spaCy Small es la elección económica para producción. LLM si presupuesto > complejidad.

---

## 3. Fuzzy Matching

### Opciones

| Librería | Algoritmo | Velocidad | Casos de uso | 
|---|---|---|---|
| **thefuzz** | Levenshtein ratio | Muy rápido (< 1ms) | OCR typos, pequeñas variaciones |
| **difflib** (stdlib) | SequenceMatcher | Muy rápido (< 1ms) | Comparaciones simples |
| **rapidfuzz** | Levenshtein (optimizado) | Ultrarrápido (SIMD) | Alto volumen |
| **Jaro-Winkler** | Distance variant | Rápido | Nombres específicamente |
| **Phonetic** (Soundex, Metaphone) | Pronunciación | Muy rápido | Solo para variaciones de pronunciación |

### Tradeoffs

**thefuzz (nuestra elección):**
- ✅ Standard Python, bien documentado
- ✅ `fuzz.ratio` compara string completo (no inflado como partial_ratio)
- ✅ Resultado normalizado 0-100
- ❌ Leve overhead comparado a difflib
- ✅ `python-Levenshtein` acelera si lo instalas

**difflib (stdlib):**
- ✅ No necesita dependencias
- ✅ Lo suficientemente rápido
- ❌ API menos intuitiva
- ❌ Resultado 0-1, no 0-100 (requiere conversión)

**rapidfuzz:**
- ✅ 10x más rápido que thefuzz
- ❌ Overkill para volumen pequeño a medio
- ❌ Dependencia extra

**Phonetic (Soundex, etc.):**
- ✅ Excelente para "Maria" vs "Mariah"
- ❌ No funciona para OCR typos (caracteres cambiados, no pronunciación)
- ❌ Requiere database de variantes

**Jaro-Winkler:**
- ✅ Diseñado específicamente para nombres
- ✅ Penaliza diferencias en inicio de string
- ❌ No mejor que Levenshtein para OCR errors
- ❌ Menos estándar

### Decisión: **thefuzz con fuzz.ratio**

**Por qué:**
- El enunciado pide "fuzzy matching" sin especificar algoritmo — Levenshtein es el default estándar
- `fuzz.ratio` (comparación completa) es correcto para nombres; `partial_ratio` era el bug original
- Umbral 90% es agrietado pero alcanzable con OCR razonable + fuzzy tolerance
- Si escalamos a 10K docs, rapidfuzz es drop-in replacement sin cambiar código

### Visualización: Fuzzy Matching — Speed vs Threshold

```
Velocidad (us/comparison)
100 │  difflib
    │     ●
 50 │
    │  thefuzz ✓
 20 │     ●
    │
 10 │  rapidfuzz
    │     ●
    │
    ├─────┬──────┬──────┬──────┬──────┬──────┤
    70%   80%    90%    95%   98%   100%  Threshold
    
  Resultado típico:
  "John Smth" (OCR typo) vs "John Smith" → 94% match ✓
  Threshold 90% acepta este match (correcto)
  Threshold 70% aceptaría falsos positivos (incorrecto)
```

**Takeaway:** `fuzz.ratio` a 90% es correcto. `partial_ratio` inflaba scores (bug original).

---

## 4. Vector Database

### Opciones

| BD | Modelo | Escalabilidad | Latencia | Características | Costo |
|---|---|---|---|---|---|
| **Qdrant** | In-memory + persistent | Media-Alta | ~50-100ms | Buenas APIs, HNSW | Gratis (open) |
| **Chroma** | Embedded SQLite | Baja-Media | ~100-200ms | Ligero, fácil setup | Gratis |
| **Milvus** | Distributed cluster | Alta | ~50ms | Enterprise-grade | Gratis (open) |
| **Weaviate** | Hybrid search | Media-Alta | ~100ms | GraphQL, multimodal | Gratis (open) |
| **Pinecone** | Cloud managed | Alta | ~50ms | Fully managed, expensive | $$$$ |
| **In-memory dict** | Python dict | Baja | ~1ms | Solo para demo | Gratis |

### Tradeoffs

**Qdrant:**
- ✅ Docker-ready (imagen oficial mantenida)
- ✅ HNSW algo (state-of-art similarity search)
- ✅ Soporta metadata filtering bien
- ✅ Buena balance entre features y complejidad
- ❌ Menos maduro que Elasticsearch
- ❌ Cluster distribution requiere setup extra

**Chroma:**
- ✅ Embebido (no necesita container separado)
- ✅ Setup trivial
- ✅ Bueno para prototipos
- ❌ SQLite backend no escala bien (> 10M vectors)
- ❌ Latencia sube con tamaño de collection
- ❌ Menos features que Qdrant

**Milvus:**
- ✅ Diseñado para escala (cluster distributed)
- ✅ Performance excelente en volumen alto
- ❌ Operación compleja (múltiples servicios: etcd, MinIO, etc.)
- ❌ Overkill para < 1M vectors
- ❌ Curva de aprendizaje steep

**Weaviate:**
- ✅ GraphQL API (flexible)
- ✅ Hybrid search (vector + keyword)
- ✅ Multimodal (text + image vectors)
- ❌ Más overhead que Qdrant
- ❌ Community menor

**Pinecone:**
- ✅ Fully managed (zero ops)
- ✅ Excelente para production without DevOps
- ❌ $$$$ por escala (billing por query + storage)
- ❌ Vendor lock-in
- ❌ No funciona offline

**In-memory dict:**
- ✅ Ultrarrápido
- ❌ Solo para demo/testing
- ❌ No persiste
- ❌ No escala

### Decisión: **Qdrant**

**Por qué:**
- El enunciado explícitamente menciona Qdrant como ejemplo
- Docker compose simplifica deployment (junto a la app)
- HNSW algo es state-of-art para similarity search
- Escala bien hasta millones de vectores (suficiente)
- Si necesitáramos escalar a 100M+ usaríamos Milvus, pero es premature optimization
- Chroma sería válida también (más ligera), pero Qdrant es más profesional

### Visualización: Vector DB — Scalability vs Ops Burden

```
Queries/segundo (throughput)
10K │                                  Pinecone
    │                                     ●
5K  │
    │                         Milvus
2K  │                           ●
    │
1K  │                  Qdrant ✓
    │                     ●
500 │          Chroma
    │            ●
    │
100 │   In-memory dict
    │      ●
    ├─────┬──────┬──────┬──────┬──────┬──────┤
  Low  Simple Docker  Cluster Distributed Enterprise
       Ops Complexity →
```

**Takeaway:** Qdrant balancea ops simplicity con performance. Chroma si zero ops. Milvus si millones de vectores.

---

## 5. Embedding Model

### Opciones

| Modelo | Parámetros | Velocidad | Accuracy | Multilingual | Tamaño |
|---|---|---|---|---|---|
| **all-MiniLM-L6-v2** | 22M | Rápido (~5ms) | Bueno | ❌ Solo EN | 22 MB |
| **all-mpnet-base-v2** | 109M | Normal (~20ms) | Muy bueno | ❌ Principalmente EN | 420 MB |
| **multilingual-e5-small** | 24M | Rápido | Bueno | ✅ 100+ idiomas | 67 MB |
| **multilingual-e5-base** | 109M | Normal | Mejor | ✅ 100+ idiomas | 438 MB |
| **Universal Sentence Encoder** | 20M | Muy rápido | OK | ✅ 16 idiomas | 55 MB |
| **OpenAI text-embedding-3-small** | Propietario | Rápido | Excelente | ✅ Multilingual | - |
| **Local LLM embeddings** (Ollama, etc.) | Varies | Lento | OK-Bueno | Varía | Varía |

### Tradeoffs

**all-MiniLM-L6-v2 (nuestra elección):**
- ✅ Ultraligero (22 MB)
- ✅ Rápido (5-10ms per query)
- ✅ Bueno accuracy para semantic search
- ✅ Estándar de facto (Hugging Face favorites)
- ❌ Solo inglés (pero OK para esta prueba)
- ✅ Open source, descarga una sola vez

**all-mpnet-base-v2:**
- ✅ Mejor accuracy que MiniLM
- ✅ Still rápido
- ❌ 10x más pesado (420 MB)
- ❌ Overkill si MiniLM resuelve

**multilingual-e5-base:**
- ✅ Múltiples idiomas
- ✅ Mejor accuracy que MiniLM
- ❌ Más pesado
- ❌ Overkill si solo inglés

**Universal Sentence Encoder:**
- ✅ Muy rápido
- ✅ Multilingual
- ❌ Accuracy mediocre
- ❌ Menos usado hoy (reemplazado por e5 y MiniLM)

**OpenAI text-embedding-3-small:**
- ✅ Excelente accuracy
- ✅ Cloud-hosted (sin overhead local)
- ❌ Costo por token ($$ a escala)
- ❌ Latencia de red
- ❌ Vendor lock-in
- ❌ No funciona offline

**Local LLM embeddings:**
- ✅ Offline, libre de costo
- ✅ Flexible
- ❌ Lento (500ms+)
- ❌ Overkill (LLM para embeddings es desperdicio)

### Decisión: **all-MiniLM-L6-v2**

**Por qué:**
- El enunciado da "Sentence-BERT" como ejemplo — ese es exactamente MiniLM
- Tamaño mínimo (22 MB) ideal para containerizar
- Velocidad crítica (embeddings deben ser < 10ms para 1000 docs/hora)
- Accuracy suficiente para RAG (la tarea es retrieval, no perfección)
- Si necesitáramos multiidioma, swap a e5-small (mismo principio)

### Visualización: Embeddings — Size vs Quality vs Speed

```
Quality (downstream task accuracy)
90 │                              OpenAI
   │                                 ●
85 │
   │              all-mpnet
80 │                 ●
   │
75 │         all-MiniLM ✓
   │            ●
70 │
   │  Universal Sentence Encoder
   │         ●
   ├─────┬──────┬──────┬──────┬──────┬──────┤
  1ms  5ms    10ms   20ms   50ms  200ms+ Latencia

Tamaño (MB):
MiniLM: 22 MB ✓
mpnet: 420 MB
OpenAI: cloud
```

**Takeaway:** MiniLM es el Goldilocks (justo, justo). mpnet si GPU. OpenAI si dinero ilimitado.

---

## 6. Web Framework

### Opciones

| Framework | Setup | Performance | Features | Community |
|---|---|---|---|---|
| **FastAPI** | Rápido | Excelente | Async, auto OpenAPI, validation | Creciente |
| **Flask** | Muy rápido | Bueno | Minimalista, flexible | Enorme |
| **Starlette** | Rápido | Excelente | Async, low-level | Pequeño |
| **Django** | Lento (setup) | Bueno | Full-featured ORM, admin | Enorme |
| **Quart** | Normal | Bueno | Async Flask | Pequeño |

### Tradeoffs

**FastAPI (nuestra elección):**
- ✅ Async nativo (mejor para I/O-bound como OCR)
- ✅ Auto OpenAPI/Swagger (sin código extra)
- ✅ Pydantic validation integrado
- ✅ Moderno, tipo hints a full
- ✅ Mejor DX (developer experience)
- ❌ Menos maduro que Flask (aunque estable)
- ✅ El enunciado sugiere "FastAPI or Flask" — FastAPI es upgrade natural

**Flask:**
- ✅ Ultraligero, minimalista
- ✅ Más maduro (community enorme)
- ✅ Fácil de aprender
- ❌ Síncrono por defecto (menos ideal para OCR)
- ❌ Menos features built-in (más boilerplate)

**Django:**
- ✅ Full-featured (ORM, admin, auth)
- ❌ Overkill para microservicio
- ❌ Setup pesado
- ❌ Más lento

### Decisión: **FastAPI**

**Por qué:**
- El enunciado da FastAPI como primera opción (con Flask como alternativa)
- Async es ventaja real para OCR (I/O-bound)
- Pydantic schemas integrados vs Flask (Flask requiere validación manual)
- OpenAPI automático = mejor documentación
- Es el estándar moderno de Python

### Visualización: Web Framework — Features vs Complexity

```
Built-in Features (# de features)
20 │                                 Django
   │                                    ●
15 │
   │                    Quart
10 │              ●     ●
   │          Starlette Flask
 5 │             ●       ●
   │                  FastAPI ✓
   │                     ●
 0 │
   ├─────┬──────┬──────┬──────┬──────┬──────┤
  1h   2h   4h   8h   16h  32h   Setup time to production

Async support:
FastAPI: ✅ native ✓
Starlette: ✅ native
Quart: ✅ native
Flask: ⚠️ workarounds
Django: ⚠️ limited
```

**Takeaway:** FastAPI = best for async I/O. Flask si amas minimalismo. Django si monolith enterprise.

---

## 7. LLM para RAG

### Opciones

| Proveedor | Costo | Latencia | Accuracy | Offline | Compliance |
|---|---|---|---|---|---|
| **OpenAI GPT-3.5** | $$ | 500ms-2s | Excelente | ❌ | ⚠️ |
| **OpenAI GPT-4** | $$$ | 1-3s | Excelente+ | ❌ | ⚠️ |
| **Anthropic Claude** | $$ | 500ms-2s | Excelente | ❌ | Mejor |
| **Google Gemini** | $$ | 1-3s | Bueno-Excelente | ❌ | ⚠️ |
| **Ollama (local)** | Gratis | 500ms-5s | OK-Bueno | ✅ | ✅ |
| **vLLM (local)** | Gratis | 100-500ms | OK-Bueno | ✅ | ✅ |
| **No LLM (extractive)** | Gratis | ~50ms | Bajo | ✅ | ✅ |

### Tradeoffs

**OpenAI GPT-3.5:**
- ✅ Excelente accuracy
- ✅ Confiable, well-tested
- ✅ Rápido (< 1s)
- ❌ Costo por token ($$)
- ❌ Data privacy concerns
- ❌ No funciona offline

**Anthropic Claude:**
- ✅ Excelente accuracy
- ✅ Mejor en razonamiento
- ✅ Buena stance en privacy
- ❌ Costo similar a OpenAI
- ❌ No funciona offline

**Ollama (local):**
- ✅ Offline, gratis, private
- ✅ Flexible (swap modelos)
- ❌ Lento (500ms-5s)
- ❌ Accuracy mediocre (llama3.2, etc.)
- ❌ Requiere compute local

**vLLM (local):**
- ✅ Más rápido que Ollama
- ✅ Offline, gratis
- ❌ Más complejo de setup
- ❌ Accuracy depende del modelo

**No LLM (extractive):**
- ✅ Ultrarrápido (50ms)
- ✅ Deterministic
- ✅ Offline
- ✅ Sin costo
- ❌ Baja quality (solo fragmentos del context)

### Decisión: **Configurable (defaultear a extractive, permitir LLM)**

**Por qué:**
- El enunciado menciona "e.g., GPT-3.5, Gemini" pero NO es obligatorio
- Nuestra implementación: el `/api/ask` funciona sin LLM (extrae fragmentos relevantes)
- Opcional: si configuras `OPENAI_API_KEY`, usa GPT
- Opcional: si configuras `OPENAI_BASE_URL=http://ollama:11434/v1`, usa Ollama local
- Esto demuestra:
  - Flexible architecture (no acoplado a un proveedor)
  - Degradación elegante (funciona aunque no haya LLM)
  - Consciencia de tradeoffs (local vs cloud, privacy vs accuracy)

---

## Matriz de Decisión (Scoring)

```
Criterio             Peso   Tesseract  spaCy  thefuzz  Qdrant  MiniLM  FastAPI  Configurable LLM
Accuracy             20%    85%        80%    90%      -       85%     -        90%
Speed                20%    95%        85%    95%      80%     90%     90%      80%
Costo                20%    100%       100%   100%     100%    100%    100%     Varía
Escalabilidad        15%    75%        70%    85%      85%     80%     90%      75%
Open source          15%    100%       100%   100%     100%    100%    100%     Mixto
DevOps simplicity    10%    80%        95%    95%      85%     95%     95%      85%
─────────────────────────────────────────────────────────────────────────────────────
Puntaje Ponderado    100%   89.5%      87%    93%      89%     89.5%   94%      86%
```

---

## Notas de Diseño

- **Tesseract + spaCy + Qdrant:** opción conservadora (bien probada, estable)
- **EasyOCR + XLM-RoBERTa + Milvus:** opción agresiva (mejor accuracy, más overhead)
- **Ollama local:** opción privacy-first (offline, gratis, flexible)

---

## Referencias

- Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
- spaCy: https://spacy.io/usage
- thefuzz: https://github.com/seatgeek/thefuzz
- Qdrant: https://qdrant.tech/documentation/
- Sentence-BERT: https://www.sbert.net/
- FastAPI: https://fastapi.tiangolo.com/
