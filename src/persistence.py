# src/persistence.py
"""Save / load trained pipeline artifacts (model + processor + clustering)."""

import json
from datetime import datetime
from pathlib import Path
import joblib
from .utils import logger

ARTIFACT_DIR = Path('models')
BUNDLE_FILE = ARTIFACT_DIR / 'pipeline.joblib'
META_FILE = ARTIFACT_DIR / 'metadata.json'


def save_artifacts(model_instance, model_name, processor, clustering,
                   metrics=None, split_info=None):
    """Persist everything needed for inference into models/."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump({
        'model': model_instance,
        'model_name': model_name,
        'processor': processor,
        'clustering': clustering,
    }, BUNDLE_FILE, compress=3)

    meta = {
        'model_name': model_name,
        'trained_at': datetime.now().isoformat(timespec='seconds'),
        'metrics': {k: float(v) for k, v in (metrics or {}).items()},
        'features': getattr(model_instance, 'feature_names_', []),
    }
    if split_info:
        meta['split'] = {
            name: {'start': str(d0)[:10], 'end': str(d1)[:10], 'rows': int(n)}
            for name, (d0, d1, n) in split_info.items()
        }
    META_FILE.write_text(json.dumps(meta, indent=2), encoding='utf-8')

    size_mb = BUNDLE_FILE.stat().st_size / 1e6
    logger.info(f'  ✓ Artifacts saved: {BUNDLE_FILE} ({size_mb:.1f} MB) + metadata.json')
    return str(BUNDLE_FILE)


def load_artifacts():
    """Load the persisted pipeline. Raises FileNotFoundError with a helpful hint."""
    if not BUNDLE_FILE.exists():
        raise FileNotFoundError(
            f'{BUNDLE_FILE} not found - train first: python main.py train')
    bundle = joblib.load(BUNDLE_FILE)
    meta = {}
    if META_FILE.exists():
        meta = json.loads(META_FILE.read_text(encoding='utf-8'))
    logger.info(f'Loaded artifacts: {bundle["model_name"]} '
                f'(trained {meta.get("trained_at", "?")})')
    return bundle, meta
