import os
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional
from openai import OpenAI
import polars as pl

# Ensure project root on sys.path when running as a script (so 'lib' is importable)
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.lib.config import load_settings

def find_openai_api_key() -> Optional[str]:
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key
    settings = load_settings()
    return settings.get("openai.api_key") or settings.get("OPENAI_API_KEY")


def read_captions(captions_path: Path) -> List[str]:
    lines: List[str] = []
    with captions_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                lines.append(s)
    return lines


def batched(seq: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def embed_texts_openai(
    texts: List[str],
    api_key: str,
    model: str = "text-embedding-3-large",
    batch_size: int = 128,
    max_retries: int = 5,
    retry_base_delay: float = 1.0,
) -> List[List[float]]:
    client = OpenAI(api_key=api_key)
    embeddings: List[List[float]] = []

    for batch in batched(texts, batch_size):
        attempt = 0
        while True:
            try:
                resp = client.embeddings.create(model=model, input=batch)
                # Order is preserved by the API
                vectors = [item.embedding for item in resp.data]
                embeddings.extend(vectors)
                break
            except Exception as e:
                print(f"Error embedding texts: {e}")
                attempt += 1
                if attempt > max_retries:
                    raise
                sleep_s = retry_base_delay * (2 ** (attempt - 1))
                time.sleep(sleep_s)

    return embeddings


def to_polars_parquet(
    texts: List[str],
    vectors: List[List[float]],
    out_path: Path,
    model: str,
) -> None:
    if len(texts) != len(vectors):
        raise ValueError("texts and vectors length mismatch")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Cast to Float32 to reduce file size; OpenAI returns float64/float32 compatible values
    vectors_f32 = [[float(x) for x in vec] for vec in vectors]

    df = pl.DataFrame(
        {
            "id": pl.arange(0, len(texts), eager=True, dtype=pl.Int64),
            "text": texts,
            "embedding": pl.Series(vectors_f32, dtype=pl.List(pl.Float32)),
        }
    ).with_columns(pl.lit(model).alias("model"))

    df.write_parquet(out_path, compression="zstd")


def main() -> None:
    settings = load_settings()

    # Inputs
    captions_path_str = (
        settings.get("captioner.embeddings.captions_file")
        or settings.get("captioner.collator.output_file")
        or str(REPO_ROOT / "captioner" / "data" / "collated_captions.txt")
    )
    captions_path = Path(captions_path_str).expanduser().resolve()

    # Outputs
    embeddings_path = (
        settings.get("captioner.embeddings.parquet")
        or settings.get("embeddings.parquet")
    )
    if not embeddings_path:
        print("Config missing key 'captioner.embeddings.parquet' (output path).", file=sys.stderr)
        sys.exit(1)
    out_path = Path(embeddings_path).expanduser().resolve()

    # API
    api_key = find_openai_api_key()
    if not api_key:
        print("OpenAI API key not found. Set OPENAI_API_KEY or add it to secrets.yml.", file=sys.stderr)
        sys.exit(1)
    model = (
        settings.get("captioner.embeddings.model")
        or settings.get("embeddings.model")
        or "text-embedding-3-large"
    )

    # Read captions
    if not captions_path.exists():
        print(f"Captions file not found: {captions_path}", file=sys.stderr)
        sys.exit(1)
    texts = read_captions(captions_path)
    if not texts:
        print("No captions found to embed.", file=sys.stderr)
        sys.exit(0)


    print(f"Embedding {len(texts)} captions with {model} ...")
    vectors = embed_texts_openai(texts, api_key=api_key, model=model, batch_size=128)
    print("Saving embeddings to Parquet ...")
    to_polars_parquet(texts, vectors, out_path, model=model)
    print(f"Done. Wrote: {out_path}")


if __name__ == "__main__":
    main()
