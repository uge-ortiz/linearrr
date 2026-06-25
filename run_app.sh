#!/bin/bash
# Lanza linear.py y abre en modo app de Chrome (sin pestañas/barra)
cd "$(dirname "$0")"

# Server en background
.venv/bin/python linear.py &
SERVER_PID=$!

# Cleanup al cerrar Chrome
trap "kill $SERVER_PID 2>/dev/null" EXIT

# Esperar a que arranque Flask
sleep 1

# Chrome en modo app (path estándar macOS)
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
if [ ! -f "$CHROME" ]; then
  echo "Chrome no encontrado en $CHROME"
  echo "Abriendo en navegador por defecto..."
  open "http://localhost:5002"
  wait $SERVER_PID
  exit 0
fi

"$CHROME" --app=http://localhost:5002 --user-data-dir="$HOME/.rrrlinear-chrome"
