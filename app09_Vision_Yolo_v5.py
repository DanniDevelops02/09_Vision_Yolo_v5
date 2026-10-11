import json

import streamlit as st
import streamlit.components.v1 as components

# ---------- Configuración de página ----------
st.set_page_config(
    page_title="Detección de Objetos en Tiempo Real",
    page_icon="🔍",
    layout="centered",
)

# ---------- Modelo ----------
# YOLOv5 (nano, 320 px) exportado a ONNX para que corra en el navegador en tiempo real.
# El archivo va en la carpeta modelos/ del repositorio.
MODEL_URL = (
    "https://raw.githubusercontent.com/DanniDevelops02/app09_Vision_Yolo_v5/main/modelos/yolov5nu.onnx"
)
TAM_MODELO = 320

# Las 80 clases de COCO, en el mismo orden del modelo, traducidas al español
CLASES = [
    "persona", "bicicleta", "carro", "moto", "avión", "bus", "tren", "camión", "barco",
    "semáforo", "hidrante", "señal de pare", "parquímetro", "banca", "pájaro", "gato",
    "perro", "caballo", "oveja", "vaca", "elefante", "oso", "cebra", "jirafa", "morral",
    "sombrilla", "bolso", "corbata", "maleta", "frisbee", "esquís", "snowboard", "balón",
    "cometa", "bate", "guante de béisbol", "patineta", "tabla de surf", "raqueta",
    "botella", "copa", "taza", "tenedor", "cuchillo", "cuchara", "tazón", "banano",
    "manzana", "sándwich", "naranja", "brócoli", "zanahoria", "perro caliente", "pizza",
    "dona", "torta", "silla", "sofá", "planta", "cama", "mesa", "inodoro", "televisor",
    "portátil", "mouse", "control remoto", "teclado", "celular", "microondas", "horno",
    "tostadora", "lavaplatos", "nevera", "libro", "reloj", "florero", "tijeras",
    "oso de peluche", "secador", "cepillo de dientes",
]

# ---------- Estilo (misma paleta que app.py) ----------
PRIMARIO = "#3f71a6"
OSCURO = "#1b2f45"
CLARO = "#eaf2f9"
SUAVE = "#a9c7e3"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"], .stApp {{
        font-family: 'Inter', sans-serif;
        font-size: 1.08rem;
    }}
    .stApp {{
        background: linear-gradient(180deg, #f6fafd 0%, {CLARO} 100%);
    }}
    .block-container {{
        max-width: 860px;
        padding-top: 3rem;
        padding-bottom: 4rem;
    }}
    .hero {{
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }}
    .hero .saludo {{
        color: {PRIMARIO};
        letter-spacing: .25em;
        text-transform: uppercase;
        font-size: .85rem;
        font-weight: 600;
    }}
    .hero h1 {{
        font-family: 'Playfair Display', serif;
        font-size: 3.2rem;
        color: {OSCURO};
        margin: .3rem 0 .6rem 0;
        line-height: 1.1;
    }}
    .hero p {{
        color: #4d6479;
        font-size: 1.2rem;
        max-width: 640px;
        margin: 0 auto;
    }}
    .linea {{
        width: 70px;
        height: 4px;
        border-radius: 4px;
        background: {PRIMARIO};
        margin: 1.4rem auto;
    }}
    h2, h3 {{
        font-family: 'Playfair Display', serif !important;
        color: {OSCURO} !important;
    }}
    [data-testid="stVerticalBlockBorderWrapper"] {{
        background: white;
        border: 1px solid {SUAVE}66 !important;
        border-radius: 18px !important;
        box-shadow: 0 8px 24px rgba(63, 113, 166, .08);
        padding: .6rem .8rem;
    }}
    [data-testid="stSidebar"] {{
        background: {OSCURO};
    }}
    [data-testid="stSidebar"] * {{
        color: #e6f0f9 !important;
    }}
    [data-testid="stSidebar"] input {{
        color: {OSCURO} !important;
        -webkit-text-fill-color: {OSCURO} !important;
    }}
    #MainMenu, footer {{ visibility: hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar ----------
with st.sidebar:
    st.subheader("Configuración de detección")
    conf_threshold = st.slider("Confianza mínima", 0.0, 1.0, 0.40, 0.01)
    iou_threshold = st.slider(
        "Umbral IoU", 0.0, 1.0, 0.45, 0.01,
        help="Qué tanto pueden superponerse dos cajas del mismo objeto antes de unirse en una.",
    )
    max_det = st.number_input("Detecciones máximas", 1, 300, 50, 1)
    st.caption("Si aparecen objetos que no están, sube la confianza. Si no detecta nada, bájala.")

# ---------- Hero ----------
st.markdown(
    """
    <div class="hero">
        <div class="saludo">Visión por computador · YOLOv5</div>
        <h1>Detección de objetos</h1>
        <p>Enciende la cámara y YOLOv5 reconocerá en tiempo real los objetos que tengas
        enfrente: personas, celulares, tazas, libros y 76 categorías más.</p>
        <div class="linea"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

config = {
    "modelUrl": MODEL_URL,
    "tam": TAM_MODELO,
    "clases": CLASES,
    "conf": conf_threshold,
    "iou": iou_threshold,
    "maxDet": int(max_det),
}

# ---------- Componente en el navegador (ONNX Runtime Web) ----------
HTML = """
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.webgpu.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@400;600&display=swap" rel="stylesheet">

<style>
  :root { --primario: __PRIMARIO__; --oscuro: __OSCURO__; --claro: __CLARO__; --suave: __SUAVE__; }
  body { margin: 0; font-family: 'Inter', sans-serif; color: var(--oscuro); }
  .wrap { display: flex; flex-direction: column; align-items: center; gap: 14px; padding: 6px; }
  .botones { display: flex; gap: 10px; flex-wrap: wrap; justify-content: center; }
  button { background: var(--primario); color: #fff; border: 0; border-radius: 999px;
           padding: .7rem 1.8rem; font-size: 1.05rem; font-weight: 600; cursor: pointer;
           font-family: inherit; transition: all .2s ease; }
  button:hover:not(:disabled) { background: var(--oscuro); transform: translateY(-2px); }
  button:disabled { opacity: .6; cursor: default; }
  button.suave { background: var(--claro); color: var(--oscuro); border: 1px solid var(--suave); }
  button.suave:hover:not(:disabled) { background: var(--primario); color: #fff; }
  .oculto { display: none; }
  #estado { font-size: .95rem; color: #4d6479; min-height: 1.2em; text-align: center; }
  .escena { position: relative; display: inline-block; max-width: 100%; }
  canvas { display: block; width: auto; height: auto; max-width: 100%; max-height: 500px;
           border-radius: 20px; background: var(--claro); box-shadow: 0 12px 32px rgba(27, 47, 69, .18); }
  #fps { position: absolute; top: 10px; right: 12px; font-size: 12px; color: #fff;
         background: rgba(27, 47, 69, .6); padding: 2px 8px; border-radius: 999px; }
  #pausa { position: absolute; top: 10px; left: 12px; font-size: 12px; color: #fff;
           background: var(--primario); padding: 2px 10px; border-radius: 999px; }
  .tabla { width: 100%; max-width: 640px; border-collapse: collapse; font-size: .95rem; }
  .tabla th { text-align: left; font-weight: 600; color: #4d6479; border-bottom: 2px solid var(--suave); padding: 6px 8px; }
  .tabla td { border-bottom: 1px solid var(--claro); padding: 6px 8px; }
  .punto { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; }
  .vacio { color: #6b8399; font-style: italic; }
</style>

<div class="wrap">
  <div class="botones">
    <button id="btn">▶ Iniciar cámara</button>
    <button id="btnPausa" class="suave oculto">⏸ Congelar imagen</button>
    <button id="btnCamara" class="suave oculto">🔄 Cambiar cámara</button>
  </div>
  <div id="estado">Al presionar el botón el navegador te pedirá permiso para usar la cámara.</div>
  <div class="escena">
    <canvas id="lienzo" width="640" height="480"></canvas>
    <div id="fps"></div>
    <div id="pausa" class="oculto">Imagen congelada</div>
  </div>
  <table class="tabla">
    <thead><tr><th>Objeto</th><th>Cantidad</th><th>Confianza promedio</th></tr></thead>
    <tbody id="cuerpo"><tr><td colspan="3" class="vacio">Aún no hay detecciones.</td></tr></tbody>
  </table>
</div>

<video id="video" playsinline muted class="oculto"></video>

<script>
const CFG = __CONFIG__;
const $ = (id) => document.getElementById(id);
const T = CFG.tam;

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";

let sesion = null, flujo = null, camara = "user";
let detecciones = [], ocupado = false, pausado = false;
let framesFps = 0, inicioFps = performance.now(), ultimaTabla = 0;

const video = $("video");
const lienzo = $("lienzo"), ctx = lienzo.getContext("2d");
const fuente = document.createElement("canvas"), fctx = fuente.getContext("2d");
const entrada = document.createElement("canvas");
entrada.width = entrada.height = T;
const ectx = entrada.getContext("2d", { willReadFrequently: true });

const color = (c) => `hsl(${(c * 47) % 360}, 75%, 50%)`;

// ---------- Modelo ----------
async function cargarModelo() {
  for (const ep of ["webgpu", "wasm"]) {
    try {
      if (ep === "webgpu" && !navigator.gpu) continue;
      sesion = await ort.InferenceSession.create(CFG.modelUrl, { executionProviders: [ep] });
      return ep;
    } catch (e) {
      console.warn("No se pudo usar", ep, e);
    }
  }
  throw new Error("no se pudo cargar el modelo desde " + CFG.modelUrl);
}

// Ajusta el cuadro a 320x320 sin deformarlo (relleno gris) y lo pasa a tensor
function preparar() {
  const W = fuente.width, H = fuente.height;
  const escala = Math.min(T / W, T / H);
  const w = Math.round(W * escala), h = Math.round(H * escala);
  const dx = (T - w) / 2, dy = (T - h) / 2;
  ectx.fillStyle = "rgb(114,114,114)";
  ectx.fillRect(0, 0, T, T);
  ectx.drawImage(fuente, dx, dy, w, h);

  const px = ectx.getImageData(0, 0, T, T).data;
  const datos = new Float32Array(3 * T * T);
  for (let i = 0, p = 0; i < T * T; i++, p += 4) {
    datos[i] = px[p] / 255;
    datos[i + T * T] = px[p + 1] / 255;
    datos[i + 2 * T * T] = px[p + 2] / 255;
  }
  return { tensor: new ort.Tensor("float32", datos, [1, 3, T, T]), escala, dx, dy };
}

function iou(a, b) {
  const x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w), y2 = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return inter / (a.w * a.h + b.w * b.h - inter);
}

// Salida [1, 84, N]: 4 valores de caja (cx, cy, w, h) + 80 puntajes de clase
function interpretar(salida, escala, dx, dy) {
  const d = salida.data, n = salida.dims[2], nc = salida.dims[1] - 4;
  const candidatas = [];
  for (let i = 0; i < n; i++) {
    let mejor = 0, clase = -1;
    for (let c = 0; c < nc; c++) {
      const s = d[(4 + c) * n + i];
      if (s > mejor) { mejor = s; clase = c; }
    }
    if (mejor < CFG.conf) continue;
    const cx = d[i], cy = d[n + i], w = d[2 * n + i], h = d[3 * n + i];
    candidatas.push({
      x: (cx - w / 2 - dx) / escala, y: (cy - h / 2 - dy) / escala,
      w: w / escala, h: h / escala, clase, conf: mejor,
    });
  }
  // Supresión de no máximos por clase
  candidatas.sort((a, b) => b.conf - a.conf);
  const finales = [];
  for (const c of candidatas) {
    if (finales.length >= CFG.maxDet) break;
    if (finales.every((f) => f.clase !== c.clase || iou(f, c) <= CFG.iou)) finales.push(c);
  }
  return finales;
}

async function detectar() {
  ocupado = true;
  try {
    const { tensor, escala, dx, dy } = preparar();
    const res = await sesion.run({ [sesion.inputNames[0]]: tensor });
    detecciones = interpretar(res[sesion.outputNames[0]], escala, dx, dy);
    framesFps++;
  } catch (e) {
    console.error(e);
  }
  ocupado = false;
}

// ---------- Dibujo ----------
function dibujar() {
  ctx.drawImage(fuente, 0, 0, lienzo.width, lienzo.height);
  const k = lienzo.width / fuente.width;
  // Letra y trazo proporcionales a la altura, para que se lean igual en vertical y horizontal
  const f = Math.max(15, Math.round(lienzo.height / 30));
  ctx.lineWidth = Math.max(3, f / 5);
  ctx.font = `600 ${f}px Inter, sans-serif`;
  for (const d of detecciones) {
    const x = d.x * k, y = d.y * k, w = d.w * k, h = d.h * k;
    ctx.strokeStyle = ctx.fillStyle = color(d.clase);
    ctx.strokeRect(x, y, w, h);
    const texto = `${CFG.clases[d.clase]} ${Math.round(d.conf * 100)}%`;
    const ancho = ctx.measureText(texto).width + f * 0.8, alto = f * 1.6;
    const ty = y > alto ? y - alto : y;
    const tx = Math.max(0, Math.min(x - ctx.lineWidth / 2, lienzo.width - ancho));
    ctx.fillRect(tx, ty, ancho, alto);
    ctx.fillStyle = "#fff";
    ctx.fillText(texto, tx + f * 0.4, ty + f * 1.15);
  }
}

function actualizarTabla() {
  const grupos = {};
  for (const d of detecciones) {
    (grupos[d.clase] = grupos[d.clase] || []).push(d.conf);
  }
  const filas = Object.entries(grupos).sort((a, b) => b[1].length - a[1].length);
  $("cuerpo").innerHTML = filas.length
    ? filas.map(([c, confs]) => `
        <tr>
          <td><span class="punto" style="background:${color(+c)}"></span>${CFG.clases[c]}</td>
          <td>${confs.length}</td>
          <td>${(confs.reduce((a, b) => a + b, 0) / confs.length).toFixed(2)}</td>
        </tr>`).join("")
    : `<tr><td colspan="3" class="vacio">No se detectan objetos con los parámetros actuales.</td></tr>`;
}

// ---------- Ciclo ----------
function ciclo() {
  if (!pausado && video.readyState >= 2) {
    // El cuadro se guarda en espejo para la cámara frontal; así cajas y textos quedan derechos
    fctx.save();
    if (camara === "user") { fctx.translate(fuente.width, 0); fctx.scale(-1, 1); }
    fctx.drawImage(video, 0, 0, fuente.width, fuente.height);
    fctx.restore();
    if (!ocupado) detectar();
  }
  dibujar();

  const ahora = performance.now();
  if (ahora - ultimaTabla > 300) { actualizarTabla(); ultimaTabla = ahora; }
  if (ahora - inicioFps >= 1000) {
    $("fps").textContent = pausado ? "" : Math.round(framesFps * 1000 / (ahora - inicioFps)) + " fps";
    framesFps = 0;
    inicioFps = ahora;
  }
  requestAnimationFrame(ciclo);
}

// ---------- Cámara ----------
async function abrirCamara() {
  if (flujo) flujo.getTracks().forEach((t) => t.stop());
  flujo = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: camara, width: { ideal: 640 }, height: { ideal: 480 } },
    audio: false,
  });
  video.srcObject = flujo;
  await video.play();
  fuente.width = video.videoWidth;
  fuente.height = video.videoHeight;
  lienzo.width = 640;
  lienzo.height = Math.round(640 * video.videoHeight / video.videoWidth);
}

async function iniciar() {
  const btn = $("btn");
  btn.disabled = true;
  try {
    $("estado").textContent = "Cargando YOLOv5…";
    const ep = await cargarModelo();

    $("estado").textContent = "Pidiendo acceso a la cámara…";
    await abrirCamara();

    // Calentamiento: la primera inferencia es lenta
    fctx.drawImage(video, 0, 0, fuente.width, fuente.height);
    await detectar();

    $("estado").textContent = `¡Listo! Detectando en tiempo real (${ep === "webgpu" ? "GPU" : "CPU"}).`;
    btn.classList.add("oculto");
    $("btnPausa").classList.remove("oculto");
    $("btnCamara").classList.remove("oculto");
    requestAnimationFrame(ciclo);
  } catch (e) {
    console.error(e);
    $("estado").textContent = "No se pudo iniciar: " + String(e.message || e).replace(/\\.+$/, "") +
      ". Revisa el permiso de cámara del navegador.";
    btn.disabled = false;
  }
}

$("btn").addEventListener("click", iniciar);

$("btnPausa").addEventListener("click", () => {
  pausado = !pausado;
  $("btnPausa").textContent = pausado ? "▶ Reanudar" : "⏸ Congelar imagen";
  $("pausa").classList.toggle("oculto", !pausado);
});

$("btnCamara").addEventListener("click", async () => {
  camara = camara === "user" ? "environment" : "user";
  try { await abrirCamara(); } catch (e) { $("estado").textContent = "Este equipo no tiene otra cámara."; }
});
</script>
"""

html = (
    HTML.replace("__CONFIG__", json.dumps(config, ensure_ascii=False))
    .replace("__PRIMARIO__", PRIMARIO)
    .replace("__OSCURO__", OSCURO)
    .replace("__CLARO__", CLARO)
    .replace("__SUAVE__", SUAVE)
)

with st.container(border=True):
    components.html(html, height=940, scrolling=False)

st.write("")

with st.container(border=True):
    st.subheader("¿Cómo funciona?")
    st.markdown(
        """
- **Modelo:** YOLOv5 nano entrenado con COCO (80 categorías), exportado a ONNX.
- **Dónde corre:** en tu navegador con ONNX Runtime Web, usando la GPU si está disponible.
  Las imágenes de la cámara no salen de tu equipo.
- **Congelar imagen:** detiene el video para revisar con calma lo que se detectó.
- **Cambiar cámara:** en el celular alterna entre la cámara frontal y la trasera.
"""
    )

st.caption("**Acerca de la aplicación**: Detección de objetos con YOLOv5 + Streamlit + ONNX Runtime Web.")
