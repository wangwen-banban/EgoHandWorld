#!/usr/bin/env bash
# Start a Cloudflare quick tunnel to the web service and allow its hostname. Runs in the foreground; Ctrl+C stops it.
#   bash ~/code/EgoHandWorld/web/public_tunnel.sh
# Anyone who has the printed https://xxx.trycloudflare.com URL can use the service (and the GPUs) while it runs.
# Quick tunnels cap each upload at 100 MB and get a new URL every start.
set -u
W=/data/wenwang/web; LOG=$W/cloudflared.log
/data/wenwang/tools/cloudflared tunnel --no-autoupdate --protocol http2 --url http://127.0.0.1:5100 > $LOG 2>&1 &
PID=$!
trap "kill $PID 2>/dev/null; sed -i \"/trycloudflare.com/d\" $W/public-host; echo; echo tunnel stopped" EXIT
for i in $(seq 1 45); do
  U=$(grep -o "https://[a-z0-9-]*\.trycloudflare\.com" $LOG | head -1)
  [ -n "$U" ] && break
  if grep -q "429\|failed" $LOG; then grep -m1 "429\|failed" $LOG; exit 1; fi
  sleep 2
done
[ -z "${U:-}" ] && { echo "no URL after 90 s; see $LOG"; exit 1; }
echo "${U#https://}" >> $W/public-host
echo "public URL: $U   (Ctrl+C to stop)"
wait $PID
