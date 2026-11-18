"""Tool paths, budgets, cost weights, and timeouts for the whole project.

Tool locations come from environment variables so that the fast test suite can run on a
machine with no LLVM toolchain. A missing tool raises ToolchainError with a message that
names the variable to set.
"""

import hashlib
import json
import os
from pathlib import Path

PROJECT_NAME = "superopt"

# Integer widths a fragment may use. Anything else is skipped by the harvester.
ALLOWED_WIDTHS = (1, 8, 16, 32, 64)

BINARY_OPS = ("add", "sub", "mul", "and", "or", "xor", "shl", "lshr", "ashr")
COMMUTATIVE_OPS = ("add", "mul", "and", "or", "xor")
SHIFT_OPS = ("shl", "lshr", "ashr")
CAST_OPS = ("zext", "sext", "trunc")
ICMP_PREDICATES = ("eq", "ne", "ugt", "uge", "ult", "ule", "sgt", "sge", "slt", "sle")
ALL_OPS = BINARY_OPS + ("icmp", "select") + CAST_OPS

# Poison-generating flags the harvester strips and records. "unknown" means LLVM reported a
# poison-generating flag that this list does not name.
KNOWN_FLAGS = ("nsw", "nuw", "exact", "disjoint", "nneg", "samesign", "unknown")

# Cost of one instruction, by opcode. Inputs and constants are free.
OP_COSTS = {
    "add": 1,
    "sub": 1,
    "and": 1,
    "or": 1,
    "xor": 1,
    "shl": 1,
    "lshr": 1,
    "ashr": 1,
    "icmp": 1,
    "select": 1,
    "mul": 3,
    "zext": 1,
    "sext": 1,
    "trunc": 0,
}

# Costs used when scoring what opt -O2 produced from an isolated fragment.
INTRINSIC_COST = 1
UNKNOWN_CALL_COST = 5
UNKNOWN_OPCODE_COST = 2
KNOWN_INTRINSICS = (
    "umin",
    "umax",
    "smin",
    "smax",
    "abs",
    "ctpop",
    "ctlz",
    "cttz",
    "bswap",
    "bitreverse",
    "fshl",
    "fshr",
)

# Fragments whose removable cost is below this are not worth synthesizing.
MIN_REMOVABLE_COST = 2

# Z3 timeouts in milliseconds.
Z3_SYNTH_TIMEOUT_MS = 2_000
Z3_VERIFY_TIMEOUT_MS = 30_000

# Alive2 SMT timeout in milliseconds.
ALIVE_SMT_TIMEOUT_MS = 20_000

# Batch sizes for external tools. One process per fragment would dominate the run time.
TOOL_BATCH_SIZE = 500
LLI_BATCH_SIZE = 200

# Targets for the llvm-mca cost check: (short name, triple, cpu).
MCA_TARGETS = (
    ("m1", "aarch64-apple-macos", "apple-m1"),
    ("skl", "x86_64-unknown-linux-gnu", "skylake"),
)

# Enumerator budgets. The expression counts decide results, so two runs agree; the wall
# clock limit is only a backstop. Measured on the development machine: about 480,000
# expressions per second on one core, so pass 1 takes about 2 s and pass 2 about 10 s.
SYNTH_EXPRESSION_BUDGET = 1_000_000
SYNTH_EXPRESSION_BUDGET_PASS2 = 5_000_000
SYNTH_WALL_SECONDS = 8.0
SYNTH_WALL_SECONDS_PASS2 = 40.0
SYNTH_BANK_CAP = 200_000
SYNTH_MAX_RESTARTS = 5
SYNTH_MAX_PROVEN = 3
# Largest number of array elements evaluated in one numpy call, to bound memory.
SYNTH_CHUNK_ELEMENTS = 2_000_000

# Number of test vectors before any counterexamples are added.
BASE_VECTOR_COUNT = 128
CARTESIAN_VECTOR_CAP = 64
VECTOR_SEED = 20261114


class ToolchainError(RuntimeError):
    """A required external tool is missing or not configured."""


def is_repo_root(path: Path) -> bool:
    """True for the directory that holds this project's pyproject.toml and harvester."""
    return (path / "pyproject.toml").is_file() and (path / "harvest" / "CMakeLists.txt").is_file()


def repo_root() -> Path:
    """Return the repository root, which holds corpus/, data/, and results/.

    The package may be installed as a copy, so the root is found from the working
    directory, not from this file: the nearest enclosing directory that looks like the
    repository. SUPEROPT_ROOT overrides the search.
    """
    override = os.environ.get("SUPEROPT_ROOT")
    if override:
        return Path(override)
    current = Path.cwd().resolve()
    for candidate in [current, *current.parents]:
        if is_repo_root(candidate):
            return candidate
    return current


def data_dir() -> Path:
    return repo_root() / "data"


def results_dir() -> Path:
    return repo_root() / "results"


def default_jobs() -> int:
    """Worker count for parallel stages: leave two cores for the writer and the system."""
    cpu_count = os.cpu_count()
    if cpu_count is None:
        return 1
    return max(1, cpu_count - 2)


def llvm_bin_dir() -> Path:
    """Return the directory that holds clang, opt, llc, lli, llvm-as, and llvm-mca."""
    value = os.environ.get("LLVM_BIN")
    if not value:
        raise ToolchainError(
            "LLVM_BIN is not set. Build the toolchain with scripts/build_toolchain.sh and "
            "source the toolchain.env file it writes."
        )
    path = Path(value)
    if not path.is_dir():
        raise ToolchainError(f"LLVM_BIN points to {path}, which is not a directory.")
    return path


def llvm_tool(name: str) -> Path:
    """Return the path of an LLVM tool such as "opt" or "llvm-mca"."""
    path = llvm_bin_dir() / name
    if not path.is_file():
        raise ToolchainError(f"LLVM tool {name} was not found at {path}.")
    return path


def alive_tv() -> Path:
    """Return the path of Alive2's alive-tv binary."""
    value = os.environ.get("ALIVE_TV")
    if not value:
        raise ToolchainError(
            "ALIVE_TV is not set. It must point to the alive-tv binary built against the "
            "same LLVM as LLVM_BIN."
        )
    path = Path(value)
    if not path.is_file():
        raise ToolchainError(f"ALIVE_TV points to {path}, which does not exist.")
    return path


def harvest_plugin() -> Path:
    """Return the path of the built harvester pass plugin."""
    override = os.environ.get("SUPEROPT_HARVEST_PLUGIN")
    if override:
        path = Path(override)
    else:
        path = repo_root() / "build" / "harvest" / "libHarvest.so"
    if not path.is_file():
        raise ToolchainError(f"Harvester plugin not found at {path}. Run `make plugin` first.")
    return path


def llvm_commit() -> str:
    """Return the pinned LLVM commit, or "unknown" when the toolchain is not configured."""
    return os.environ.get("LLVM_COMMIT", "unknown")


def alive2_commit() -> str:
    """Return the Alive2 commit, or "unknown" when the toolchain is not configured."""
    return os.environ.get("ALIVE2_COMMIT", "unknown")


def toolchain_available() -> bool:
    """True when the LLVM tools and alive-tv can be found. Used to skip toolchain tests."""
    try:
        llvm_tool("opt")
        alive_tv()
    except ToolchainError:
        return False
    return True


def config_hash(settings: dict) -> str:
    """Hash a stage's settings so cached results are reused only under the same settings."""
    text = json.dumps(settings, sort_keys=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
