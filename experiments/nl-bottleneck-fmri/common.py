"""Shared pieces of the fMRI text-bottleneck experiment.

The models come from the story-listening fMRI run on all TRs (runs/neuro/fmri-may27-run1), at the
iterations reported in the hidden-set table of the paper. Scoring repeats that table's procedure:
ridge regression is fit on the run's 8 fitting stories with the run's own feature and ridge code
(all TRs), then evaluated on the 2 public visible stories and on 10 hidden stories.

Paths: DATA holds the public release files (TextGrids, per-story responses, respdict.json and
split.json); WORK holds sandboxes and caches. Both are set with environment variables.
"""
import hashlib, importlib.util, json, os, re, sys, types
from pathlib import Path
import numpy as np
import h5py

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUN = REPO / 'experiments' / 'runs' / 'neuro' / 'fmri-may27-run1'
DATA = Path(os.environ.get('NLBF_DATA', '/path/to/story_data'))
WORK = Path(os.environ.get('NLBF_WORK', '/path/to/work'))
os.environ.setdefault('NLTK_DATA', str(DATA / 'nltk'))
sys.path.insert(0, str(RUN))
from src import features, encoding  # the run's own feature pipeline and ridge regression

BAD = frozenset(['sentence_start', 'sentence_end', 'br', 'lg', 'ls', 'ns', 'sp'])
TR, START, PAD = 2.0, 10.0, 5
MODELS = [  # (iteration, shorthand name) from the hidden-set table
    (1, 'BoCharMean-d256'), (40, 'MultiScaleSubword+FuncWordTags'), (76, 'AllNewFeats-Words15'),
    (152, 'ExpDec-R4.0'), (192, 'DenseSemXX-N20'), (238, 'WordNetMorphLingMultiTau'),
    (239, 'WordNetMorphLingStoryArc'), (242, 'WordNetMorphLingNovelty')]


class Seq:
    pass


def split(data=DATA):
    return json.loads((data / 'split.json').read_text())


def wordseq(story, data=DATA):
    respdict = json.loads((data / 'respdict.json').read_text())
    text = (data / 'grids' / f'{story}.TextGrid').read_text(errors='replace')
    tier = text[text.index('name = "word"'):]
    entries = [(float(a), float(b), t.lower()) for a, b, t in
               re.findall(r'xmin = ([\d.e+-]+)\s*\n\s*xmax = ([\d.e+-]+)\s*\n\s*text = "(.*?)"', tier)
               if t.strip() and t.lower().strip('{}').strip() not in BAD]
    seq = Seq()
    seq.data = np.array([t for *_, t in entries])
    seq.data_times = np.array([(a + b) / 2 for a, b, _ in entries])
    seq.tr_times = np.arange(respdict[story] - PAD) * TR - START + TR / 2
    return seq


def response(story, data=DATA):
    with h5py.File(data / 'resp' / f'{story}.hf5', 'r') as f:
        d = f['data'][()].astype(np.float32)
    return np.nanmean(d, axis=0) if d.ndim == 3 else d


def load_embedder(path, name=None):
    """Import a model file and return its embedder: build_embedder() if defined, else its *Embedder class."""
    if 'src.eval' not in sys.modules:  # model files import the run harness at module level; it is never called here
        stub = types.ModuleType('src.eval')
        for k in ('EncodingConfig', 'run_encoding', 'make_result_row', 'upsert_overall_results', 'plot_corr_over_iterations'):
            setattr(stub, k, None)
        sys.modules['src.eval'] = stub
    path = Path(path)
    spec = importlib.util.spec_from_file_location(name or 'm_' + re.sub(r'\W', '_', path.stem), path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(mod)
    if hasattr(mod, 'build_embedder'):
        try:
            return mod.build_embedder(device='cpu')
        except TypeError:
            return mod.build_embedder()
    classes = [v for k, v in vars(mod).items() if k.endswith('Embedder') and isinstance(v, type) and v.__module__ == mod.__name__]
    assert len(classes) == 1, f'{path}: expected one *Embedder class, found {[c.__name__ for c in classes]}'
    return classes[0]()


def fit_and_predict(embed, fit_stories, eval_stories, data=DATA):
    seqs = {s: wordseq(s, data) for s in fit_stories + eval_stories}
    Y_fit = np.nan_to_num(np.vstack([response(s, data) for s in fit_stories]))
    X_fit = features.get_features(seqs, fit_stories, embed)
    parts = [features.get_features({s: seqs[s]}, [s], embed) for s in eval_stories]
    X_eval = np.vstack(parts)
    Y_eval = np.nan_to_num(np.vstack([response(s, data) for s in eval_stories]))
    fit = encoding.fit_encoding(X_fit, Y_fit, X_eval, Y_eval)
    pred = np.nan_to_num(X_eval @ fit['weights']).astype(np.float32)
    edges = np.cumsum([0] + [len(p) for p in parts])
    spans = {s: np.arange(edges[i], edges[i + 1]) for i, s in enumerate(eval_stories)}
    return dict(train_corr=float(fit['corrs_train_mean']), n_features=int(X_fit.shape[1]), pred=pred, Y=Y_eval, spans=spans)


def voxel_corr(y, p):
    return encoding._per_voxel_corr(y, p)
