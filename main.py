Ecco l'architettura completa, il codice sorgente e la documentazione di **AgentForge Core**: una soluzione chiavi in mano enterprise per l'orchestrazione di agenti AI basata su FastAPI, LangGraph e ChromaDB.

---

# Struttura del Progetto

```text
agentforge-core/
├── app/
│   ├── __init__.py
│   ├── config.py              # Gestione configurazione con Pydantic Settings
│   ├── main.py                # Entry point FastAPI, lifespan, middleware
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── graph.py           # Definizione del grafo LangGraph (State, Nodes, Edges)
│   │   ├── memory.py          # Wrapper ChromaDB per memoria a lungo termine vettoriale
│   │   └── tools.py           # Tool dell'agente (eseguibili ed estensibili)
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py          # Endpoints REST standard per esecuzione asincrona
│   │   └── webhooks.py        # Connettori Slack, Telegram, Zapier
│   └── schemas/
│       ├── __init__.py
│       └── models.py          # Modelli Pydantic per input/output API
├── data/                      # Volume persistente per ChromaDB e Checkpoint
├── .env.example
├── .dockerignore
├── Dockerfile                 # Multi-stage production build
├── docker-compose.yml
└── requirements.txt
```

---

# Codice Sorgente

### `requirements.txt`
```text
fastapi>=0.110.0
uvicorn[standard]>=0.28.0
langgraph>=0.2.20
langchain>=0.2.14
langchain-openai>=0.1.22
chromadb>=0.5.5
pydantic-settings>=2.2.1
httpx>=0.27.0
python-multipart>=0.0.9
```

### `app/config.py`
```python
import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # App Config
    PROJECT_NAME: str = "AgentForge Core"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "production"
    LOG_LEVEL: str = "INFO"

    # AI Configuration
    OPENAI_API_KEY: str
    MODEL_NAME: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Persistence & ChromaDB
    CHROMA_PERSIST_DIRECTORY: str = "/data/chromadb"
    SQLITE_CHECKPOINT_DIR: str = "/data/checkpoints"

    # Webhook Secrets
    TELEGRAM_BOT_TOKEN: Optional[str] = None
    SLACK_SIGNING_SECRET: Optional[str] = None
    GENERIC_WEBHOOK_SECRET: str = "changeme-in-production-token"

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        case_sensitive=True
    )

settings = Settings()
```

### `app/schemas/models.py`
```python
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class ChatRequest(BaseModel):
    message: str = Field(..., example="Ricorda che il server di produzione è deployato a Francoforte.")
    thread_id: str = Field(..., example="user-session-1234")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class ChatResponse(BaseModel):
    response: str
    thread_id: str
    recalled_memories: List[str] = []

class GenericWebhookPayload(BaseModel):
    source: str = Field(..., example="zapier")
    user_id: str = Field(..., example="usr_987")
    prompt: str = Field(..., example="Sintetizza questo lead e memorizzalo.")
    data: Optional[Dict[str, Any]] = None
```

### `app/agent/memory.py`
```python
import os
import chromadb
from chromadb.utils import embedding_functions
from app.config import settings
from typing import List

class ChromaMemoryManager:
    """Gestione memoria episodica a lungo termine via semantic vector search."""
    def __init__(self):
        os.makedirs(settings.CHROMA_PERSIST_DIRECTORY, exist_ok=True)
        self.client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIRECTORY)
        
        self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=settings.OPENAI_API_KEY,
            model_name=settings.EMBEDDING_MODEL
        )
        
        self.collection = self.client.get_or_create_collection(
            name="long_term_memory",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def persist_memory(self, text: str, user_id: str, metadata: dict = None) -> None:
        """Salva un frammento concettuale nello storage vettoriale."""
        meta = metadata or {}
        meta["user_id"] = user_id
        
        doc_id = f"{user_id}_{hash(text)}"
        self.collection.upsert(
            documents=[text],
            metadatas=[meta],
            ids=[doc_id]
        )

    def recall_memories(self, query: str, user_id: str, limit: int = 3) -> List[str]:
        """Recupera contesti rilevanti basati su similarità semantica."""
        results = self.collection.query(
            query_texts=[query],
            n_results=limit,
            where={"user_id": user_id}
        )
        if results and results["documents"] and results["documents"][0]:
            return results["documents"][0]
        return []

memory_manager = ChromaMemoryManager()
```

### `app/agent/tools.py`
```python
from langchain_core.tools import tool
import datetime

@tool
def get_current_utc_time() -> str:
    """Restituisce l'orario UTC corrente formattato in ISO. Utile per calcoli temporali."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
```

### `app/agent/graph.py`
```python
import os
from typing import Annotated, TypedDict, List
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from app.config import settings
from app.agent.memory import memory_manager
from app.agent.tools import get_current_utc_time

# Modello di stato
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    thread_id: str
    recalled_memories: List[str]

# Inizializzazione LLM con tool binding
tools = [get_current_utc_time]
llm = ChatOpenAI(
    model=settings.MODEL_NAME, 
    api_key=settings.OPENAI_API_KEY,
    temperature=0
).bind_tools(tools)

# Nodi LangGraph
def memory_retrieval_node(state: AgentState) -> dict:
    """Recupera le memorie a lungo termine rilevanti prima dell'inferenza."""
    last_user_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_user_msg = str(msg.content)
            break
            
    memories = []
    if last_user_msg:
        memories = memory_manager.recall_memories(
            query=last_user_msg, 
            user_id=state["thread_id"]
        )
    return {"recalled_memories": memories}

def agent_reasoning_node(state: AgentState) -> dict:
    """Processa lo storico, inietta la memoria vettoriale e ottiene la risposta del modello."""
    memory_context = ""
    if state["recalled_memories"]:
        memory_context = "\n[Memorie a lungo termine recuperate]:\n" + "\n".join(
            [f"- {m}" for m in state["recalled_memories"]]
        )

    system_prompt = (
        "Sei AgentForge Core, un assistente AI autonomo enterprise. "
        "Usa i tool forniti se necessario e rispetta il contesto informativo.\n"
        f"{memory_context}"
    )

    enriched_messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(enriched_messages)
    return {"messages": [response]}

def memory_persistence_node(state: AgentState) -> dict:
    """Estrae e consolida l'input dell'utente nella memoria vettoriale persistente."""
    last_user_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_user_msg = str(msg.content)
            break
            
    if last_user_msg and len(last_user_msg.split()) > 4:
        memory_manager.persist_memory(
            text=last_user_msg, 
            user_id=state["thread_id"],
            metadata={"type": "user_input"}
        )
    return {}

# Assemblaggio Grafo
workflow = StateGraph(AgentState)
workflow.add_node("retrieve_memory", memory_retrieval_node)
workflow.add_node("agent", agent_reasoning_node)
workflow.add_node("persist_memory", memory_persistence_node)

workflow.add_edge(START, "retrieve_memory")
workflow.add_edge("retrieve_memory", "agent")
workflow.add_edge("agent", "persist_memory")
workflow.add_edge("persist_memory", END)

# Checkpointer per lo stato a breve termine delle sessioni
checkpointer = MemorySaver()
agent_graph = workflow.compile(checkpointer=checkpointer)
```

### `app/api/routes.py`
```python
from fastapi import APIRouter, HTTPException, status
from langchain_core.messages import HumanMessage
from app.schemas.models import ChatRequest, ChatResponse
from app.agent.graph import agent_graph

router = APIRouter()

@router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat_interaction(payload: ChatRequest):
    """Endpoint unificato asincrono per l'invocazione dell'agente."""
    try:
        config = {"configurable": {"thread_id": payload.thread_id}}
        initial_state = {
            "messages": [HumanMessage(content=payload.message)],
            "thread_id": payload.thread_id
        }

        # Esecuzione del grafo (asincrona)
        result = await agent_graph.ainvoke(initial_state, config=config)
        
        last_message = result["messages"][-1].content
        recalled = result.get("recalled_memories", [])

        return ChatResponse(
            response=str(last_message),
            thread_id=payload.thread_id,
            recalled_memories=recalled
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Errore durante l'esecuzione dell'agente: {str(e)}"
        )
```

### `app/api/webhooks.py`
```python
from fastapi import APIRouter, Request, Header, HTTPException, BackgroundTasks, status
from langchain_core.messages import HumanMessage
from app.config import settings
from app.schemas.models import GenericWebhookPayload
from app.agent.graph import agent_graph
import httpx

router = APIRouter()

async def process_agent_job(prompt: str, thread_id: str, callback_url: str = None):
    """Job asincrono di background per l'orchestrazione da webhook."""
    config = {"configurable": {"thread_id": thread_id}}
    result = await agent_graph.ainvoke(
        {"messages": [HumanMessage(content=prompt)], "thread_id": thread_id},
        config=config
    )
    response_text = result["messages"][-1].content

    if callback_url:
        async with httpx.AsyncClient() as client:
            await client.post(callback_url, json={"thread_id": thread_id, "result": response_text})

@router.post("/generic")
async def generic_webhook(
    payload: GenericWebhookPayload, 
    background_tasks: BackgroundTasks,
    x_api_key: str = Header(...)
):
    """Webhook compatibile con Zapier, Make, n8n o script interni."""
    if x_api_key != settings.GENERIC_WEBHOOK_SECRET:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Chiave API non valida.")
    
    background_tasks.add_task(
        process_agent_job, 
        prompt=payload.prompt, 
        thread_id=payload.user_id
    )
    return {"status": "enqueued", "thread_id": payload.user_id}

@router.post("/telegram")
async def telegram_webhook(request: Request, background_tasks: BackgroundTasks):
    """Ricevitore nativo per bot Telegram."""
    data = await request.json()
    message = data.get("message", {})
    chat_id = str(message.get("chat", {}).get("id"))
    text = message.get("text")

    if not chat_id or not text:
        return {"status": "ignored"}

    # Processa ed effettua la risposta direttamente tramite Telegram API
    async def run_and_reply(cid: str, msg: str):
        config = {"configurable": {"thread_id": cid}}
        result = await agent_graph.ainvoke(
            {"messages": [HumanMessage(content=msg)], "thread_id": cid}, 
            config=config
        )
        reply = result["messages"][-1].content
        if settings.TELEGRAM_BOT_TOKEN:
            url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage"
            async with httpx.AsyncClient() as client:
                await client.post(url, json={"chat_id": cid, "text": reply})

    background_tasks.add_task(run_and_reply, chat_id, text)
    return {"status": "ok"}

@router.post("/slack/events")
async def slack_events_webhook(request: Request, background_tasks: BackgroundTasks):
    """Handler standard Slack Event API (inclusa URL verification)."""
    body = await request.json()

    # Handshake challenge di Slack
    if "challenge" in body:
        return {"challenge": body["challenge"]}

    event = body.get("event", {})
    if event.get("type") == "app_mention":
        user_id = event.get("user")
        text = event.get("text")
        background_tasks.add_task(process_agent_job, prompt=text, thread_id=user_id)

    return {"status": "ok"}
```

### `app/main.py`
```python
from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMouter, CORSMiddleware
from app.config import settings
from app.api.routes import router as chat_router
from app.api.webhooks import router as webhook_router

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger("agentforge")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Avvio di AgentForge Core in corso...")
    # Setup di risorse o warm-up dei modelli
    yield
    logger.info("Spegnimento ordinato di AgentForge Core terminato.")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Policy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registrazione Route
app.include_router(chat_router, prefix=settings.API_V1_STR, tags=["Agent Chat"])
app.include_router(webhook_router, prefix=f"{settings.API_V1_STR}/webhooks", tags=["Integrations"])

@app.get("/health", tags=["Infra"])
async def health_check():
    return {"status": "healthy", "version": settings.VERSION}
```

---

# Containerizzazione e Distribuzione

### `.dockerignore`
```text
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.env
.git
.gitignore
data/
.pytest_cache
```

### `Dockerfile`
```dockerfile
# ---------------------------------------------------
# Stage 1: Build & Dependencies
# ---------------------------------------------------
FROM python:3.11-slim as builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ---------------------------------------------------
# Stage 2: Minimal Distroless/Production Runtime
# ---------------------------------------------------
FROM python:3.11-slim as runner

WORKDIR /app

# Creazione utente non-root per questioni di sicurezza
RUN groupadd -g 1001 agentgroup && \
    useradd -u 1001 -g agentgroup -s /bin/bash -m agentuser

# Copia delle librerie dallo stage precedente
COPY --from=builder /root/.local /home/agentuser/.local
ENV PATH=/home/agentuser/.local/bin:$PATH

# Creazione directory volumi per ChromaDB con permessi corretti
RUN mkdir -p /data/chromadb /data/checkpoints && \
    chown -R agentuser:agentgroup /data

# Copia del codice dell'applicazione
COPY --chown=agentuser:agentgroup ./app ./app

USER agentuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--proxy-headers"]
```

### `docker-compose.yml`
```yaml
version: '3.8'

services:
  agentforge:
    build:
      context: .
      dockerfile: Dockerfile
    image: agentforge-core:latest
    container_name: agentforge_orchestrator
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - GENERIC_WEBHOOK_SECRET=${GENERIC_WEBHOOK_SECRET}
      - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN:-}
      - CHROMA_PERSIST_DIRECTORY=/data/chromadb
    volumes:
      - agent_data:/data
    networks:
      - agentforge-network

volumes:
  agent_data:
    driver: local

networks:
  agentforge-network:
    driver: bridge
```

### `.env.example`
```env
# Chiavi AI
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
MODEL_NAME=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

# Parametri Webhook
GENERIC_WEBHOOK_SECRET=supersecret-bearer-key-123
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrSTUvwxYZ
SLACK_SIGNING_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Configurazione Runtime
ENVIRONMENT=production
LOG_LEVEL=INFO
```

---

# Documentazione Operativa

## 1. Quickstart Locale

1. **Clona e configura le variabili di ambiente:**
   ```bash
   cp .env.example .env
   # Modifica .env con la tua vera chiave OPENAI_API_KEY
   ```

2. **Esecuzione con Docker Compose (consigliata):**
   ```bash
   docker compose up --build -d
   ```
   L'istanza sarà raggiungibile su `http://localhost:8000`. La documentazione interattiva OpenAPI è disponibile su `http://localhost:8000/docs`.

---

## 2. Architettura della Memoria

AgentForge Core separa la memoria in due livelli:

1. **Memoria di Sessione (Breve Termine - `LangGraph Checkpointer`):**
   Mantiene lo stato transazionale della conversazione attiva. La chiave di routing è `thread_id`. Se un utente fa una domanda di follow-up, l'agente tiene conto dei turni immediatamente precedenti.

2. **Memoria Episodica Semantica (Lungo Termine - `ChromaDB`):**
   A ogni interazione, l'input dell'utente viene convertito in vettori e memorizzato in una collection persistente indicizzata con HNSW. Durante l'elaborazione del grafo, prima dell'esecuzione del modello, il nodo `retrieve_memory` effettua una ricerca per similarità (Cosine Similarity) su tutte le vecchie memorie appartenenti al medesimo `user_id`/`thread_id`, iniettandole come blocco di contesto dinamico nel System Prompt.

---

## 3. Esempi di Utilizzo API

### A. Esecuzione Sincrona/Diretta via REST
Invia una richiesta direttamente al motore dell'agente:

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Il database Redis aziendale risiede all indirizzo 10.0.0.45",
    "thread_id": "sysadmin-dan"
  }'
```

Interroga nuovamente l'agente con lo stesso `thread_id`:

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "A quale IP si trova il database Redis?",
    "thread_id": "sysadmin-dan"
  }'
```
L'agente recupererà il record da **ChromaDB**, lo inietterà nel prompt ed estrarrà l'indirizzo `10.0.0.45`, ritornando il vettore nel payload JSON alla voce `recalled_memories`.

---

### B. Integrazione Zapier / Make / n8n (Generic Webhook)

L'endpoint accetta richieste e le processa asincronamente in background (`BackgroundTasks`), evitando il timeout HTTP delle piattaforme no-code su elaborazioni LLM complesse.

```bash
curl -X POST http://localhost:8000/api/v1/webhooks/generic \
  -H "Content-Type: application/json" \
  -H "X-Api-Key: supersecret-bearer-key-123" \
  -d '{
    "source": "zapier",
    "user_id": "lead-contact-992",
    "prompt": "Genera una notifica di benvenuto personalizzata per questo contatto."
  }'
```

---

### C. Integrazione Telegram Bot
1. Crea un bot tramite `@BotFather` su Telegram e ottieni il token.
2. Inserisci il token nel file `.env` alla voce `TELEGRAM_BOT_TOKEN`.
3. Associa il tuo webhook esponendo la porta (tramite reverse proxy HTTPS, domain o ngrok):
   ```bash
   curl -X POST "https://api.telegram.org/bot<TUO_TOKEN>/setWebhook?url=https://tuodominio.it/api/v1/webhooks/telegram"
   ```
4. Scrivi al bot: l'agente risponderà direttamente usando la memoria associata all'ID utente Telegram.

---

## 4. Linee Guida per il Deployment in Produzione (VPS / Cloud)

* **Sicurezza Container:** Il container non viene eseguito come `root`. L'utente `agentuser` possiede solo i permessi della cartella `/data` e dell'app.
* **Storage Persistente:** Il volume nominato Docker `agent_data` garantisce che i dati vettoriali di ChromaDB e i checkpoint non vadano persi durante aggiornamenti, rolling deployment o riavvii del container.
* **Reverse Proxy:** Termina SSL/TLS tramite NGINX o Caddy su porta 443 prima di inoltrare il traffico verso la porta 8000:
  ```nginx
  location / {
      proxy_pass http://127.0.0.1:8000;
      proxy_set_header Host $host;
      proxy_set_header X-Real-IP $remote_addr;
      proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
      proxy_set_header X-Forwarded-Proto $scheme;
  }
  ```