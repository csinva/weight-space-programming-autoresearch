"""Load the Huth fMRI language dataset (wordseqs + fMRI responses).

Simplified from `neuro/data/{stim_utils,response_utils,story_names}.py`.
Uses the pre-packaged "huge_data" joblib files, which are the fastest to load
and are already aligned: each story's response is trimmed by [10:-5] relative
to its TR timeline (so downsampled features must be trimmed identically).
"""
import os
import sys
from os.path import join

import joblib
import numpy as np

# the cached wordseqs were pickled referencing the top-level `ridge_utils`
# package, so make our shim importable before any joblib.load call
sys.path.insert(0, os.path.dirname(__file__))
import ridge_utils  # noqa: E402,F401

FMRI_DIR = '/home/chansingh/mntv1/deep-fMRI/data'
HUGE_DIR = join(FMRI_DIR, 'huge_data')
# named anatomical/functional ROIs: {roi_name: voxel_index_array} into the voxel space
REGION_IDXS_DIR = '/home/chansingh/mntv1/deep-fMRI/sasc/brain_regions'

# per-story response cache (read-only mntv1 is never modified). Caching avoids
# re-reading the ~26GB full-subject response file on every iteration of the loop.
CACHE_DIR = os.path.expanduser('~/.cache/evolve-lang-fmri')

# the training stories shared across the UTS01/UTS02/UTS03 "huge" training sets
# (intersection of all three), plus the standard held-out test stories. Using the
# shared set keeps the same stimuli usable for any of the three subjects.
TRAIN_STORIES = [
    'itsabox', 'odetostepfather', 'inamoment', 'hangtime', 'ifthishaircouldtalk',
    'goingthelibertyway', 'golfclubbing', 'thetriangleshirtwaistconnection',
    'igrewupinthewestborobaptistchurch', 'tetris', 'becomingindian',
    'canplanetearthfeedtenbillionpeoplepart1', 'thetiniestbouquet', 'swimmingwithastronauts',
    'lifereimagined', 'forgettingfear', 'stumblinginthedark', 'backsideofthestorm', 'food',
    'theclosetthatateeverything', 'notontheusualtour', 'exorcism', 'adventuresinsayingyes',
    'thefreedomridersandme', 'cocoonoflove', 'waitingtogo', 'thepostmanalwayscalls',
    'googlingstrangersandkentuckybluegrass', 'mayorofthefreaks', 'learninghumanityfromdogs',
    'shoppinginchina', 'souls', 'cautioneating', 'comingofageondeathrow',
    'breakingupintheageofgoogle', 'gpsformylostidentity', 'eyespy', 'treasureisland',
    'thesurprisingthingilearnedsailingsoloaroundtheworld', 'theadvancedbeginner',
    'goldiethegoldfish', 'life', 'thumbsup', 'seedpotatoesofleningrad', 'theshower', 'adollshouse',
    'canplanetearthfeedtenbillionpeoplepart2', 'sloth', 'howtodraw', 'quietfire', 'metsmagic',
    'penpal', 'thecurse', 'canadageeseandddp', 'thatthingonmyarm', 'buck',
    'wildwomenanddancingqueens', 'againstthewind', 'indianapolis', 'alternateithicatom', 'bluehope',
    'kiksuya', 'afatherscover', 'haveyoumethimyet', 'firetestforlove',
    'catfishingstrangerstofindmyself', 'christmas1940', 'tildeath', 'lifeanddeathontheoregontrail',
    'vixenandtheussr', 'undertheinfluence', 'beneaththemushroomcloud', 'jugglingandjesus',
    'superheroesjustforeachother', 'sweetaspie', 'naked', 'singlewomanseekingmanwich', 'avatar',
    'whenmothersbullyback', 'myfathershands', 'reachingoutbetweenthebars', 'theinterview',
    'stagefright', 'legacy', 'canplanetearthfeedtenbillionpeoplepart3', 'listo',
    'gangstersandcookies', 'birthofanation', 'mybackseatviewofagreatromance',
    'lawsthatchokecreativity', 'threemonths', 'whyimustspeakoutaboutclimatechange',
    'leavingbaghdad',
]
TEST_STORIES = ['fromboyhoodtofatherhood', 'wheretheressmoke', 'onapproachtopluto']


def get_story_names(num_train=None, num_test=1):
    """Return (train_stories, test_stories)."""
    train = TRAIN_STORIES if num_train is None else TRAIN_STORIES[:num_train]
    return train, TEST_STORIES[:num_test]


def load_wordseqs(stories):
    """Return {story: DataSequence} with .data (words), .data_times, .tr_times."""
    wordseqs = joblib.load(join(HUGE_DIR, 'wordseqs.joblib'))
    return {s: wordseqs[s] for s in stories}


def load_responses(stories, subject='UTS03'):
    """Return {story: (n_trs, n_voxels)} fMRI responses (already trimmed [10:-5]).

    Responses are cached per story under CACHE_DIR; the big ~26GB subject file is
    only read when some requested story is not yet cached.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    def cache_path(story):
        return join(CACHE_DIR, f'{subject}_{story}.jbl')

    out = {}
    missing = [s for s in stories if not os.path.exists(cache_path(s))]
    if missing:
        resps = joblib.load(join(HUGE_DIR, f'{subject}_responses.jbl'))
        for s in missing:
            joblib.dump(resps[s], cache_path(s))
        del resps
    for s in stories:
        out[s] = joblib.load(cache_path(s))
    return out


def load_rois(subject='UTS03'):
    """Return {roi_name: voxel_index_array} for a subject (empty dict if none on disk).

    Indices index into the subject's voxel axis (same axis as the responses).
    Only UTS02/UTS03 have ROI files.
    """
    s = subject.replace('UT', '')  # 'UTS03' -> 'S03'
    path = join(REGION_IDXS_DIR, f'rois_{s}.jbl')
    if not os.path.exists(path):
        return {}
    return {k: np.asarray(v) for k, v in joblib.load(path).items()}
