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
