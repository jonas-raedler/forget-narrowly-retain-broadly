#!/usr/bin/env bash
# Pod finalize hook: publish results before the pod is stopped/terminated.
# (Adapted verbatim from tamper_resistant_unlearning/finalize_pod.sh — see
# that file's header for the credential-scoping rationale: identity and auth
# are process-scoped env only, nothing is written to .git/config.)
#
# Exit non-zero on a real publish failure — remote-kernels then degrades
# terminate to stop so the results stay collectable.
set -euo pipefail
cd "$(dirname "$0")"

export GIT_AUTHOR_NAME="${GIT_AUTHOR_NAME:-runpod-finalize}"
export GIT_AUTHOR_EMAIL="${GIT_AUTHOR_EMAIL:-runpod-finalize@users.noreply.github.com}"
export GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME"
export GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"

if [ -n "${GITHUB_TOKEN:-}" ]; then
  B64=$(printf 'x-access-token:%s' "$GITHUB_TOKEN" | base64 | tr -d '\n')
  export GIT_CONFIG_COUNT=1
  export GIT_CONFIG_KEY_0="http.https://github.com/.extraHeader"
  export GIT_CONFIG_VALUE_0="Authorization: Basic ${B64}"
fi

bash publish_results.sh
