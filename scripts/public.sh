#!/usr/bin/env bash
set -euo pipefail

ORIGIN_URL=${ORIGIN_URL:-http://127.0.0.1:${PORT:-18320}}
CLOUDFLARED_BIN=${CLOUDFLARED_BIN:-cloudflared}
curl --noproxy '*' -fsS --max-time 5 "$ORIGIN_URL/healthz" >/dev/null

# HTTP/2 uses TCP 443 and avoids depending on outbound UDP availability.
exec "$CLOUDFLARED_BIN" tunnel --no-autoupdate --protocol http2 --url "$ORIGIN_URL"
