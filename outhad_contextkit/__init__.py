try:
    import importlib.metadata
    __version__ = importlib.metadata.version("outhad_contextkitai")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.1.0-dev"

from outhad_contextkit.client.main import AsyncMemoryClient, MemoryClient  # noqa
from outhad_contextkit.memory.context_graph import ContextGraphConfig, EdgeType  # noqa
from outhad_contextkit.memory.main import AsyncMemory, Memory  # noqa
