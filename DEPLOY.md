# Guía de despliegue — AsistenteTrading en EasyPanel (Hostinger VPS)

## Paso 1 — Crear repositorio en GitHub

1. Ve a **github.com** e inicia sesión (o crea una cuenta gratis)
2. Haz click en **"New repository"**
3. Nombre: `asistente-trading` (o el que prefieras)
4. Marca **"Private"** (para que nadie más vea tu código)
5. Haz click en **"Create repository"**

## Paso 2 — Subir el proyecto a GitHub

Abre la terminal en VS Code (en la carpeta del proyecto) y ejecuta:

```bash
git init
git add .
git commit -m "Primer commit: AsistenteTrading MVP"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/asistente-trading.git
git push -u origin main
```

> Reemplaza `TU_USUARIO` con tu nombre de usuario de GitHub.

## Paso 3 — Configurar EasyPanel en tu VPS Hostinger

1. Entra al panel de Hostinger y accede a tu **VPS**
2. Abre **EasyPanel** (normalmente en el puerto 3000 de tu VPS)
3. Haz click en **"Create Project"**
4. Nombre del proyecto: `trading-bot`

## Paso 4 — Crear los dos servicios en EasyPanel

### Servicio 1: Bot (scheduler)

1. Dentro del proyecto, haz click en **"+ Create Service"** → **"App"**
2. Selecciona **"GitHub"** y conecta tu cuenta
3. Elige el repo `asistente-trading`
4. En **"Build"**: selecciona `Dockerfile`
5. En **"Start Command"**: `python scheduler/main.py`
6. En **"Environment Variables"** agrega (ver sección de Variables):
   - `EXCHANGE_API_KEY`
   - `EXCHANGE_API_SECRET`
   - `EXCHANGE_TESTNET = true`
   - `TELEGRAM_TOKEN`
   - `TELEGRAM_CHAT_ID`
   - `TELEGRAM_ENABLED = false` (cambiar a true cuando tengas el token)
   - `TRADING_MODE = paper`
   - `TRADING_CAPITAL = 1000`
   - `PYTHONPATH = /app`
7. Haz click en **"Deploy"**

### Servicio 2: Dashboard (web)

1. **"+ Create Service"** → **"App"**
2. Mismo repo `asistente-trading`
3. En **"Start Command"**:
   ```
   python -m streamlit run dashboard/app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true
   ```
4. En **"Ports"**: exponer el puerto `8501`
5. En **"Domain"**: EasyPanel te dará una URL pública (ej: `trading.tudominio.com`)
6. Mismas Variables de entorno que el Servicio 1
7. Haz click en **"Deploy"**

## Paso 5 — Primer entrenamiento en el VPS

Una vez que el bot esté corriendo en el VPS, necesitas hacer el primer entrenamiento.
Conéctate al VPS por SSH y ejecuta:

```bash
docker exec -it <nombre_del_contenedor_bot> python run.py
```

O desde EasyPanel, abre la **Terminal** del servicio bot y ejecuta:
```bash
python run.py
```
Elige la opción **1** para entrenar el modelo.

## Paso 6 — Acceder desde el celular

- **Dashboard:** Abre el navegador de tu celular y entra a la URL que te dio EasyPanel (ej: `https://trading.tudominio.com`)
- **Telegram:** Configura el bot de Telegram siguiendo las instrucciones dentro del dashboard

## Variables de entorno — Referencia completa

| Variable | Descripción | Ejemplo |
|----------|-------------|---------|
| `EXCHANGE_API_KEY` | API Key de Binance | `abc123...` |
| `EXCHANGE_API_SECRET` | API Secret de Binance | `xyz789...` |
| `EXCHANGE_TESTNET` | Usar testnet (prueba) | `true` |
| `TELEGRAM_TOKEN` | Token del bot de Telegram | `1234567:ABC...` |
| `TELEGRAM_CHAT_ID` | Tu Chat ID de Telegram | `123456789` |
| `TELEGRAM_ENABLED` | Activar notificaciones | `true` |
| `TRADING_MODE` | Modo de trading | `paper` o `live` |
| `TRADING_CAPITAL` | Capital inicial simulado (USD) | `1000` |

## Auto-deploy (opcional)

Cada vez que hagas `git push` al repo, EasyPanel puede redesplegar automáticamente.
Activa **"Auto Deploy"** en la configuración de cada servicio en EasyPanel.

## IMPORTANTE: Antes de activar dinero real

1. El bot debe correr en **paper trading mínimo 2-3 meses**
2. Revisa el dashboard regularmente desde tu celular
3. El Win Rate debe ser consistentemente superior al 52%
4. Solo entonces cambia `TRADING_MODE=live` en EasyPanel
5. Obtén las API Keys de Binance con **solo permisos de trading** (sin retiros)
