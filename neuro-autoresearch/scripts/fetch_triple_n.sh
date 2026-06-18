#!/usr/bin/env bash
# Fetch the Triple-N (NNN) **Processed** files from ScienceDB onto this machine.
#
# Triple-N (DOI 10.57760/sciencedb.33556, dataset 413ba7ddcb694bb2a534270caa90be36)
# is macaque visual-cortex spiking to the NSD-1000 images. The full release is ~2.7 TB
# of raw spikes/LFP; we only want the compact **Processed** per-session matrices
# ([units x 1072 images] firing rates), which are all the table-loader interface needs.
#
# ScienceDB is LOGIN-GATED: every file/listing API returns "User Not Login" anonymously
# (verified by reverse-engineering their Nuxt API). So this script does NOT log in for you.
# Instead it consumes the direct-download manifest that ScienceDB hands a logged-in user.
#
# ── How to get the manifest (one manual step, in a browser where you're logged in) ──
#   1. Open https://www.scidb.cn/detail?dataSetId=413ba7ddcb694bb2a534270caa90be36
#   2. Log in, then click the dataset's "Download" → "Copy/Download all URLs" button.
#      That yields an `all_url.txt`: one direct, pre-tokenized download URL per line.
#   3. Put that file on this machine (scp/paste) and pass it as $MANIFEST below.
#      The URLs already carry an access token, so wget needs no further auth.
#
# Usage:
#   MANIFEST=/path/to/all_url.txt scripts/fetch_triple_n.sh          # download Processed only
#   MANIFEST=all_url.txt PATTERN='Processed|GoodUnit' scripts/fetch_triple_n.sh
#   MANIFEST=all_url.txt DRYRUN=1 scripts/fetch_triple_n.sh          # just list what matches
set -euo pipefail

DEST="${DEST:-$HOME/Research/neuralset/data/nnn_processed}"
MANIFEST="${MANIFEST:?set MANIFEST=/path/to/all_url.txt (see header for how to get it)}"
PATTERN="${PATTERN:-[Pp]rocessed}"   # which files to pull; default = Processed-only

[ -f "$MANIFEST" ] || { echo "manifest not found: $MANIFEST" >&2; exit 1; }
mkdir -p "$DEST"

# Keep only http(s) lines matching the wanted pattern.
mapfile -t urls < <(grep -E '^https?://' "$MANIFEST" | grep -E "$PATTERN" || true)
n=${#urls[@]}
echo "manifest: $MANIFEST"
echo "matched $n file(s) against /$PATTERN/ -> $DEST"
[ "$n" -gt 0 ] || { echo "no URLs matched; widen PATTERN or check the manifest." >&2; exit 1; }
printf '  %s\n' "${urls[@]}" | sed -E 's|\?.*$||' | head -40
[ "$n" -gt 40 ] && echo "  ... ($((n-40)) more)"

[ -n "${DRYRUN:-}" ] && { echo "(dry run; nothing downloaded)"; exit 0; }

# Resumable, content-disposition-aware download. ScienceDB serves the real filename
# via Content-Disposition, so -P + --content-disposition lands them named correctly.
printf '%s\n' "${urls[@]}" | wget --content-disposition -c -P "$DEST" \
    --tries=3 --timeout=60 -i -

echo
echo "done. Staged under: $DEST"
echo "next: inspect one .mat to confirm the Processed schema, then wire the 'nnn' table loader."
