#!/usr/bin/env bash
# Publish experiment result artifacts: commit the small evaluations/ outputs
# (nllOutputs, worstCase, evalJudge, mmluOutputs, results_db.json — see
# .gitignore carve-outs) to main and push. Generations/logs/checkpoints stay
# on the pod volume.
#
# Retry-safe like the tamper_resistant_unlearning original: only exits early
# when the index is clean AND nothing is ahead of origin/main, so a prior
# commit that failed to push still gets pushed on retry.
set -euo pipefail
cd "$(dirname "$0")/.."

git add evaluations 2>/dev/null || true

# PUBLIC-REPO SAFETY GATE: this runs unattended on the pod and pushes to a
# PUBLIC fork. Refuse to commit if anything staged looks like a credential
# (HF tokens, GitHub PATs, private keys). Exit non-zero → finalize_pod.sh
# fails → remote-kernels degrades terminate to stop, data stays private.
if git diff --cached | grep -qE 'hf_[A-Za-z0-9]{30,}|ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|gho_[A-Za-z0-9]{30,}|-----BEGIN[A-Z ]*PRIVATE KEY-----|AKIA[0-9A-Z]{16}'; then
  echo "ERROR: staged results contain a credential-like string — refusing to publish." >&2
  git reset -q
  exit 1
fi

if ! git diff --cached --quiet; then
  CODE_SHA="$(git rev-parse --short HEAD)"
  git commit -m "results: proximity sweep outputs (code @ ${CODE_SHA})"
fi

git fetch origin main
if [ "$(git rev-list --count origin/main..HEAD)" -eq 0 ]; then
  echo "Nothing to publish (index clean, HEAD not ahead of origin/main)."
  exit 0
fi

git pull --rebase --autostash origin main
git push origin main
echo "Published results to origin/main."
