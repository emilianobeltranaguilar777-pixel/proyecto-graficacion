#!/usr/bin/env bash
# Reinicia v4l2loopback, ADB forward y lanza droidcam-cli -> /dev/video${VID_IDX}
# Uso: recover_droidcam_usb.sh [VID_IDX] (default: 2)
set -euo pipefail

VID_IDX="${1:-2}"
DEV="/dev/video${VID_IDX}"

# Rutas absolutas
MODPROBE="/usr/sbin/modprobe"
V4L2CTL="/usr/bin/v4l2-ctl"
ADB="/usr/bin/adb"
DROIDCAM="/usr/local/bin/droidcam-cli"

# Directorios seguros para PID/LOG (systemd los crea si usamos RuntimeDirectory/LogsDirectory)
RUNDIR="/run/droidcam"
LOGDIR="/var/log/droidcam"
mkdir -p "$RUNDIR" "$LOGDIR"
chmod 0755 "$RUNDIR" "$LOGDIR"

PIDF="${RUNDIR}/droidcam.pid"
LOG="${LOGDIR}/droidcam.log"

for bin in "$MODPROBE" "$V4L2CTL" "$ADB" "$DROIDCAM"; do
  [[ -x "$bin" ]] || { echo "[ERR] Falta binario: $bin" >&2; exit 1; }
done

echo "[INFO] Preparando v4l2loopback en ${DEV} (label: DroidCam, exclusive_caps=1)"

# Reset módulo
"$MODPROBE" -r v4l2loopback || true
"$MODPROBE" v4l2loopback "video_nr=${VID_IDX}" exclusive_caps=1 'card_label=DroidCam'

# Esperar /dev/videoN
echo -n "[INFO] Esperando a ${DEV} "
for _ in {1..50}; do
  [[ -e "$DEV" ]] && { echo "-> OK"; break; }
  echo -n "."; sleep 0.1
done
[[ -e "$DEV" ]] || { echo; echo "[ERR] No apareció ${DEV}"; exit 1; }

# Fijar formato base (no crítico si falla)
"$V4L2CTL" -d "$DEV" --set-fmt-video=width=640,height=480,pixelformat=YU12 || true

# ADB forward
echo "[INFO] Iniciando ADB y configurando forward tcp:4747"
"$ADB" start-server >/dev/null
"$ADB" forward --remove-all || true
"$ADB" forward tcp:4747 tcp:4747 >/dev/null

# Relanzar feeder
echo "[INFO] Matando feeders previos (si los hay)"
pkill -f droidcam-cli || true

echo "[INFO] Lanzando droidcam-cli -> ${DEV}"
# Redirigir logs a LOGDIR y guardar PID en RUNDIR
# shellcheck disable=SC2086
nohup "$DROIDCAM" 127.0.0.1 4747 -size 640x480 -fps 30 -dev="$DEV" >>"$LOG" 2>&1 &
echo $! > "$PIDF"

sleep 0.2
if ps -p "$(cat "$PIDF")" >/dev/null 2>&1; then
  echo "[OK] DroidCam alimentando ${DEV} (PID $(cat "$PIDF"))."
  echo "[INFO] Log: $LOG"
else
  echo "[ERR] droidcam-cli no quedó ejecutándose. Revisa $LOG" >&2
  exit 1
fi
