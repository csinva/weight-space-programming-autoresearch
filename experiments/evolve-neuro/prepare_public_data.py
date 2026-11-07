"""Build the story-listening fMRI data from the public release (OpenNeuro ds003020).

Downloads each story's preprocessed responses and word timings, then writes what src/data.py
reads: huge_data/wordseqs.joblib, huge_data/<subject>_responses.jbl and huge_data/stories.txt.
Set FMRI_DIR to choose the data root. Downloads about 3 GB for the default stories.

    uv run experiments/evolve-neuro/prepare_public_data.py --subject UTS03
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from os.path import dirname, join

import h5py
import joblib
import numpy as np

sys.path.insert(0, join(dirname(os.path.abspath(__file__)), 'src'))
import data  # noqa: E402
from ridge_utils.DataSequence import DataSequence  # noqa: E402

BUCKET = 'https://s3.amazonaws.com/openneuro.org/ds003020/derivatives'
SKIPPED_WORDS = frozenset(['sentence_start', 'sentence_end', 'br', 'lg', 'ls', 'ns', 'sp'])
TR, SOUND_START, PAD = 2.0, 10.0, 5
RELEASE_TRIM = 15  # TRs the responses are shorter than the simulated TR timeline


def fetch(url, path):
    if not os.path.exists(path):
        subprocess.run(['curl', '-sSfL', '--retry', '5', '--retry-all-errors', '-o', path, url], check=True)
    return path


def word_sequence(grid_path, n_trs):
    """Words and times from a story's TextGrid, on a TR timeline simulated from its length."""
    with open(grid_path, errors='replace') as f:
        tier = f.read().split('name = "word"')[1]
    pattern = r'xmin = ([\d.e+-]+)\s*\n\s*xmax = ([\d.e+-]+)\s*\n\s*text = "(.*?)"'
    entries = [(float(a), float(b), text.lower()) for a, b, text in re.findall(pattern, tier)
               if text.strip() and text.lower().strip('{}').strip() not in SKIPPED_WORDS]
    words = np.array([text for _, _, text in entries])
    word_times = np.array([(a + b) / 2 for a, b, _ in entries])
    tr_times = np.arange(n_trs - PAD) * TR - SOUND_START + TR / 2
    return DataSequence(words, None, data_times=word_times, tr_times=tr_times)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--subject', default='UTS03', choices=['UTS01', 'UTS02', 'UTS03'])
    parser.add_argument('--num-train', type=int, default=8, help='First N shared training stories.')
    parser.add_argument('stories', nargs='*', help='Extra stories to include, such as a hidden set.')
    args = parser.parse_args()

    raw = join(data.FMRI_DIR, 'public_release')
    for folder in (join(raw, 'grids'), join(raw, 'resp'), data.HUGE_DIR):
        os.makedirs(folder, exist_ok=True)
    lengths = json.load(open(fetch(f'{BUCKET}/respdict.json', join(raw, 'respdict.json'))))

    wanted = data.TRAIN_STORIES[:args.num_train] + data.TEST_STORIES + args.stories
    stories = [s for s in dict.fromkeys(wanted) if s in lengths]
    print('not in the public release:', sorted(set(wanted) - set(stories)))

    wordseqs, responses = {}, {}
    for story in stories:
        print(story, flush=True)
        grid = fetch(f'{BUCKET}/TextGrids/{story}.TextGrid', join(raw, 'grids', f'{story}.TextGrid'))
        path = fetch(f'{BUCKET}/preprocessed_data/{args.subject}/{story}.hf5', join(raw, 'resp', f'{story}.hf5'))
        with h5py.File(path, 'r') as f:
            responses[story] = np.nan_to_num(f['data'][()].astype(np.float32))
        wordseqs[story] = word_sequence(grid, lengths[story])
        assert len(wordseqs[story].tr_times) - RELEASE_TRIM == len(responses[story]), story

    joblib.dump(wordseqs, join(data.HUGE_DIR, 'wordseqs.joblib'))
    joblib.dump(responses, join(data.HUGE_DIR, f'{args.subject}_responses.jbl'))
    with open(join(data.HUGE_DIR, 'stories.txt'), 'w') as f:
        f.write('\n'.join(stories) + '\n')
    print(f'wrote {len(stories)} stories to {data.HUGE_DIR}')


if __name__ == '__main__':
    main()
