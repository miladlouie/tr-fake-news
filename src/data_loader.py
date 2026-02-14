from pathlib import Path
import pandas as pd
from .config import DATA_PATH


def load_data(dataset_path=None):
    """
    Supports:
    1) CSV dataset → text,label
    2) Folder dataset:
         base/Fake/*.txt → label 0
         base/Real/*.txt → label 1
    """

    if dataset_path is None:
        dataset_path = DATA_PATH

    path = Path(dataset_path)

    # ---------- CASE 1: CSV ----------
    if path.is_file():
        print(f"Loading CSV dataset: {path}")
        df = pd.read_csv(path)

        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError(
                f"CSV must contain columns: text,label. Found: {df.columns.tolist()}"
            )

        return df.dropna(subset=["text", "label"])

    # ---------- CASE 2: Folder dataset ----------
    if path.is_dir():
        fake_dir = path / "Fake"
        real_dir = path / "Real"

        if not fake_dir.exists() or not real_dir.exists():
            raise ValueError(
                f"Folder dataset must contain Fake/ and Real/ directories inside {path}"
            )

        rows = []

        # Load FAKE
        for file in fake_dir.glob("*.txt"):
            try:
                text = file.read_text(encoding="utf-8", errors="ignore").strip()
                if text:
                    rows.append({"text": text, "label": 0})
            except Exception:
                pass

        # Load REAL
        for file in real_dir.glob("*.txt"):
            try:
                text = file.read_text(encoding="utf-8", errors="ignore").strip()
                if text:
                    rows.append({"text": text, "label": 1})
            except Exception:
                pass

        df = pd.DataFrame(rows)

        print(f"Loaded folder dataset:")
        print(f"  Fake: {len(list(fake_dir.glob('*.txt')))} files")
        print(f"  Real: {len(list(real_dir.glob('*.txt')))} files")
        print(f"  Usable rows: {len(df)}")

        if len(df) == 0:
            raise ValueError("No valid text files found in dataset folders")

        return df

    raise ValueError(f"Dataset path not found: {dataset_path}")
