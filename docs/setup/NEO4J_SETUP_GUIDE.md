# Neo4j Setup Guide for TCMGM

Complete guide to setting up Neo4j for Outhad_ContextKit's TCMGM features.

---

## What is TCMGM?

**TCMGM** (Temporal-Causal-Multimodal Graph Memory) enables:
- ⏰ Temporal reasoning ("What changed before the bug?")
- 🔗 Causal analysis ("Why did this happen?")
- 📅 Timeline queries ("What happened last week?")
- 🕸️ Entity relationship graphs
- 🎯 Fused retrieval (vector + graph + timeline)

**Requires Neo4j** graph database.

---

## Option 1: Neo4j Desktop (Recommended for Development)

### Step 1: Download

1. Go to [neo4j.com/download](https://neo4j.com/download/)
2. Select "Neo4j Desktop"
3. Download for your OS (Mac/Windows/Linux)
4. Install the application

### Step 2: Install

- **Mac**: Open DMG, drag to Applications
- **Windows**: Run installer
- **Linux**: Extract and run

### Step 3: Create Database

1. Open Neo4j Desktop
2. Click **"+ New"** → **"Create Project"**
3. Name your project: "Outhad_ContextKit"
4. Click **"Add"** → **"Local DBMS"**
5. **Name**: "outhad_contextkit"
6. **Password**: Choose a strong password (remember this!)
7. **Version**: 5.x (latest)
8. Click **"Create"**

### Step 4: Start Database

1. Click **"Start"** on your database
2. Wait for status to show **"Active"**
3. Note the connection URI: `bolt://localhost:7687`

### Step 5: Configure Outhad_ContextKit

```python
from outhad_contextkit import Memory
from outhad_contextkit.configs.base import MemoryConfig
from outhad_contextkit.graphs.configs import Neo4jConfig, GraphStoreConfig

config = MemoryConfig(
    graph_store=GraphStoreConfig(
        provider="neo4j",
        config=Neo4jConfig(
            url="bolt://localhost:7687",
            username="neo4j",
            password="your-password",  # Password you set
            database="neo4j"
        )
    )
)

memory = Memory(config=config)
```

### Step 6: Verify Connection

```python
# Test TCMGM is enabled
results = memory.search(
    query="test",
    user_id="test_user",
    use_tcmgm=True  # Should work now
)

print("✅ TCMGM enabled and working!")
```

---

## Option 2: Docker (Recommended for Production)

### Step 1: Install Docker

- **Mac**: [Docker Desktop](https://www.docker.com/products/docker-desktop)
- **Linux**: `sudo apt install docker.io`
- **Windows**: [Docker Desktop](https://www.docker.com/products/docker-desktop)

### Step 2: Run Neo4j Container

```bash
docker run \
    --name neo4j-outhad \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/your-password \
    -v $HOME/neo4j/data:/data \
    -d \
    neo4j:latest
```

**Parameters explained**:
- `--name neo4j-outhad`: Container name
- `-p 7474:7474`: Browser interface port
- `-p 7687:7687`: Bolt protocol port
- `-e NEO4J_AUTH`: Set username/password
- `-v`: Persist data to disk
- `-d`: Run in background

### Step 3: Verify Running

```bash
# Check if running
docker ps | grep neo4j

# Should show:
# neo4j-outhad   neo4j:latest   Up 2 minutes   0.0.0.0:7474->7474/tcp, 0.0.0.0:7687->7687/tcp

# Access browser interface
open http://localhost:7474
```

### Step 4: Configure Environment Variables

```bash
# Add to ~/.bashrc or ~/.zshrc
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="your-password"

# Reload shell
source ~/.bashrc
```

### Step 5: Configure Outhad_ContextKit

```python
import os
from outhad_contextkit import Memory
from outhad_contextkit.configs.base import MemoryConfig
from outhad_contextkit.graphs.configs import Neo4jConfig, GraphStoreConfig

config = MemoryConfig(
    graph_store=GraphStoreConfig(
        provider="neo4j",
        config=Neo4jConfig(
            url=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD"),
            database="neo4j"
        )
    )
)

memory = Memory(config=config)
```

### Docker Management Commands

```bash
# Start container
docker start neo4j-outhad

# Stop container
docker stop neo4j-outhad

# View logs
docker logs neo4j-outhad

# Remove container (data persists)
docker rm neo4j-outhad

# Remove data (careful!)
rm -rf $HOME/neo4j/data
```

---

## Option 3: Neo4j Aura (Managed Cloud)

### Step 1: Sign Up

1. Go to [neo4j.com/cloud/aura](https://neo4j.com/cloud/aura/)
2. Create free account
3. Verify email

### Step 2: Create Instance

1. Click **"Create Instance"**
2. Select **"AuraDB Free"**
3. **Name**: "outhad-contextkit"
4. **Region**: Choose closest to you
5. Click **"Create"**

### Step 3: Save Credentials

⚠️ **IMPORTANT**: Save credentials when shown - they're only displayed once!

```
Connection URI: neo4j+s://xxxxx.databases.neo4j.io
Username: neo4j
Password: [generated password]
```

**Save these immediately!**

### Step 4: Configure Outhad_ContextKit

```python
from outhad_contextkit import Memory
from outhad_contextkit.configs.base import MemoryConfig
from outhad_contextkit.graphs.configs import Neo4jConfig, GraphStoreConfig

config = MemoryConfig(
    graph_store=GraphStoreConfig(
        provider="neo4j",
        config=Neo4jConfig(
            url="neo4j+s://xxxxx.databases.neo4j.io",  # Your URI
            username="neo4j",
            password="generated-password",  # Your password
            database="neo4j"
        )
    )
)

memory = Memory(config=config)
```

### Step 5: Test Connection

```python
# Add a memory with TCMGM
memory.add(
    "This is a test event",
    user_id="test_user",
    metadata={"type": "test"}
)

# Search with TCMGM
results = memory.search(
    query="test",
    user_id="test_user",
    use_tcmgm=True
)

print(f"✅ Found {len(results['results'])} results")
```

---

## Troubleshooting

### Connection Refused

**Problem**: `neo4j.exceptions.ServiceUnavailable: Failed to establish connection`

**Solution**:
```bash
# Desktop: Check if database is started
# Docker: docker ps | grep neo4j
# Aura: Check instance status in console

# For Docker, start container:
docker start neo4j-outhad

# Verify it's running:
curl http://localhost:7474
```

### Authentication Failed

**Problem**: `neo4j.exceptions.AuthError: Authentication failed`

**Solution**:
- **Desktop**: Check password in Neo4j Desktop settings
- **Docker**: Verify NEO4J_AUTH environment variable
- **Aura**: Retrieve credentials from Aura console (if lost, create new instance)

### Database Not Found

**Problem**: `Database 'neo4j' not found`

**Solution**:
```python
# Use system database (always exists)
config = Neo4jConfig(
    url="bolt://localhost:7687",
    username="neo4j",
    password="your-password",
    database="system"  # or "neo4j"
)
```

### Port Already in Use

**Problem**: Docker error: `port 7687 already allocated`

**Solution**:
```bash
# Find what's using the port
lsof -i :7687

# Stop existing Neo4j
docker stop neo4j-outhad

# Or use different ports
docker run -p 7475:7474 -p 7688:7687 ...
```

---

## Comparison

| Feature | Desktop | Docker | Aura |
|---------|---------|--------|------|
| **Setup Time** | 10 min | 5 min | 5 min |
| **Best For** | Development | Production | Production |
| **Cost** | Free | Free | Free tier available |
| **Data Persistence** | Local | Local | Cloud |
| **Scalability** | Limited | Medium | High |
| **Maintenance** | Manual | Manual | Managed |

---

## Recommendations

- **Development**: Use **Neo4j Desktop** for GUI and easy management
- **Docker Users**: Use **Docker** for consistent environments
- **Production**: Use **Docker** or **Aura** for reliability
- **Quick Start**: Use **Aura** if you don't want to manage infrastructure

---

## Next Steps

Once Neo4j is set up:

1. **[TCMGM User Guides](../TCMGM/)** - Learn TCMGM features
2. **[Timeline Helper Guide](../TCMGM/PHASE4_TIMELINE_USER_GUIDE.md)** - Timeline queries
3. **[Retrieval Orchestrator Guide](../TCMGM/PHASE5_RETRIEVAL_ORCHESTRATOR_USER_GUIDE.md)** - Fused search
4. **[Examples](../../examples/)** - Real-world applications

---

