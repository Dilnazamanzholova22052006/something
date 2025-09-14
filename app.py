import streamlit as st
# --- PyTorch 2.6 safe-load FIX (делай это только если доверяешь своим .pt) ---
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel, SegmentationModel, PoseModel, ClassificationModel

# Стандартные torch.nn классы, которые часто встречаются в чекпоинтах
import torch.nn as nn

add_safe_globals([
    # Ultralytics модели
    DetectionModel, SegmentationModel, PoseModel, ClassificationModel,

    # Популярные модули из torch.nn
    nn.Sequential, nn.ModuleList, nn.ReLU, nn.SiLU, nn.LeakyReLU, nn.Sigmoid,
    nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, nn.GroupNorm,
    nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.Upsample,
    nn.Linear, nn.Dropout,
])
# --- конец FIX ---

from ultralytics import YOLO
from PIL import Image
import cv2
import io

# ===== Page & Theme =====
st.set_page_config(page_title="Car Condition Analyzer", page_icon="🚗", layout="wide")

# ---- Minimal CSS for clean UI ----
st.markdown("""
<style>
/* Global tweaks */
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
/* Fancy title bar */
.hero {
  background: linear-gradient(135deg, #111827 0%, #1f2937 50%, #0ea5e9 100%);
  color: white; padding: 22px 22px; border-radius: 18px; margin-bottom: 18px;
  border: 1px solid rgba(255,255,255,0.06);
}
/* Badges */
.badge {display:inline-block; padding:6px 10px; border-radius:999px; font-weight:600; font-size:0.9rem; margin-right:8px;}
.badge-green {background:#10b98122; color:#10b981; border:1px solid #10b98155;}
.badge-red {background:#ef444422; color:#ef4444; border:1px solid #ef444455;}
.badge-blue {background:#3b82f622; color:#3b82f6; border:1px solid #3b82f655;}
/* Cards */
.card {border:1px solid rgba(0,0,0,0.08); border-radius:16px; padding:16px;}
/* Section titles */
h3, h4 {margin-top: 0.6rem;}
/* Tips */
.tip {font-size:0.9rem; opacity:0.85}
</style>
""", unsafe_allow_html=True)

# ---- Header ----
st.markdown("""
<div class="hero">
  <h1 style="margin:0;">🚗 Car Condition Analyzer</h1>
  <div class="tip">Загрузи фото машины — получишь аккуратный вывод по чистоте и повреждениям с аннотациями.</div>
</div>
""", unsafe_allow_html=True)

# ---- Sidebar ----
with st.sidebar:
    st.header("⚙️ Options")
    st.caption("UX-настройки отображения (логика модели не меняется).")
    show_dirty_layer = st.checkbox("Показывать слой 'Dirty'", value=True)
    show_damage_layer = st.checkbox("Показывать слой 'Damage'", value=True)
    st.divider()
    st.subheader("ℹ️ Help")
    with st.expander("Как это работает?"):
        st.write(
            "- Загружаешь JPG/PNG.\n"
            "- Первая модель отмечает **dirty** (грязь). Если нет боксов класса dirty — считаем чистой.\n"
            "- Вторая модель отмечает **damaged** (повреждения). Если боксов нет — считаем без повреждений.\n"
            "\n⚠️ Мы не меняем пороги/логику — только оформление."
        )

# >>> вставь СРАЗУ после import-ов, ПЕРЕД YOLO("...") <<<
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel, SegmentationModel, PoseModel, ClassificationModel
add_safe_globals([DetectionModel, SegmentationModel, PoseModel, ClassificationModel])
# <<< конец фикса >>>

from ultralytics import YOLO

# дальше как было:
model_clean_dirty = YOLO("models/dirty_best.pt")
model_damage     = YOLO("models/damaged2_best.pt")
# ======= Load models (paths must exist) =======
@st.cache_resource(show_spinner=True)
def load_models():
    m_cd = YOLO("models/dirty_best.pt")       # 2 класса: clean(0), dirty(1)
    m_dmg = YOLO("models/damaged2_best.pt")   # 1 класс: damaged(0)
    return m_cd, m_dmg

try:
    model_clean_dirty, model_damage = load_models()
    model_status = True
except Exception as e:
    model_status = False
    st.error(f"❌ Не удалось загрузить модели: {e}")

# ======= Core logic (UNCHANGED) =======
def analyze_car(image):
    # clean/dirty (только проверяем, есть ли dirty)
    result_cd = model_clean_dirty.predict(image, verbose=False)
    if len(result_cd[0].boxes) > 0:
        # проверяем есть ли хоть один dirty
        dirty_found = any(int(cls.item()) == 1 for cls in result_cd[0].boxes.cls)
        if dirty_found:
            cd_label = "This car is dirty!"
        else:
            cd_label = "This car is clean!"
    else:
        cd_label = "This car is clean!"

    # damage/no damage (покажем боксы)
    result_dmg = model_damage.predict(image, verbose=False)
    if len(result_dmg[0].boxes) > 0:
        dmg_label = "This car is damaged!"
    else:
        dmg_label = "This car is not damaged!"

    return cd_label, dmg_label, result_cd, result_dmg

# ======= Uploader =======
uploaded_file = st.file_uploader("📤 Перетащи сюда изображение или выбери файл", type=["jpg", "jpeg", "png"])

if not model_status:
    st.stop()

if uploaded_file is None:
    st.info("👆 Загрузите фото, чтобы начать. Рекомендация: снимок сбоку/¾, хорошее освещение.")
else:
    image = Image.open(uploaded_file).convert("RGB")

    # Left: preview; Right: results
    col_left, col_right = st.columns([7, 5], gap="large")

    with col_left:
        st.subheader("🖼️ Просмотр")
        st.image(image, caption="Оригинал", use_container_width=True, output_format="PNG")

    with col_right:
        st.subheader("🔍 Результаты анализа")
        with st.spinner("Анализируем изображение…"):
            clean_dirty, damage, result_cd, result_dmg = analyze_car(image)

        # Badges summary
        badge_cd = ('<span class="badge badge-red">Dirty</span>'
                    if "dirty" in clean_dirty.lower()
                    else '<span class="badge badge-green">Clean</span>')
        badge_dmg = ('<span class="badge badge-red">Damaged</span>'
                     if "damaged" in damage.lower()
                     else '<span class="badge badge-green">No Damage</span>')

        st.markdown(f"""
        <div class="card">
          <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap;">
            <span class="badge badge-blue">Summary</span>
            {badge_cd}{badge_dmg}
          </div>
          <div style="margin-top:8px;">
            <b>Clean/Dirty:</b> {clean_dirty}<br/>
            <b>Damage:</b> {damage}
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ======= Tabs for annotated views =======
    tabs = st.tabs(["🧩 Вкладки аннотаций", "💾 Скачивание", "❔ Подсказки"])

    with tabs[0]:
        subcol1, subcol2 = st.columns(2)
        # Clean/Dirty Detection (рисуем только dirty)
        with subcol1:
            st.markdown("#### Грязь (dirty)")
            if show_dirty_layer and "dirty" in clean_dirty.lower():
                cd_annotated = result_cd[0].orig_img.copy()
                for box in result_cd[0].boxes:
                    cls = int(box.cls[0].item())
                    if cls == 1:  # dirty
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(cd_annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(cd_annotated, "dirty", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                cd_annotated = cv2.cvtColor(cd_annotated, cv2.COLOR_BGR2RGB)
                st.image(cd_annotated, caption="📌 Clean/Dirty Detection", use_container_width=True)
            else:
                st.info("Нет отмеченных областей грязи или слой отключён.")

        # Damage Detection
        with subcol2:
            st.markdown("#### Повреждения (damaged)")
            if show_damage_layer and ("damaged" in damage.lower()):
                dmg_annotated = result_dmg[0].plot()
                dmg_annotated = cv2.cvtColor(dmg_annotated, cv2.COLOR_BGR2RGB)
                st.image(dmg_annotated, caption="📌 Damage Detection", use_container_width=True)
            else:
                st.info("Нет отмеченных повреждений или слой отключён.")

    with tabs[1]:
        st.markdown("#### Сохранить результаты")
        # Буферы для скачивания
        dl_cols = st.columns(2)

        # Prepare annotated dirty image if exists
        dirty_img_bytes = None
        if "dirty" in clean_dirty.lower():
            cd_annotated = result_cd[0].orig_img.copy()
            for box in result_cd[0].boxes:
                cls = int(box.cls[0].item())
                if cls == 1:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(cd_annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(cd_annotated, "dirty", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            cd_annotated = cv2.cvtColor(cd_annotated, cv2.COLOR_BGR2RGB)
            buf = io.BytesIO()
            Image.fromarray(cd_annotated).save(buf, format="PNG")
            dirty_img_bytes = buf.getvalue()

        # Prepare annotated damage image if exists
        damage_img_bytes = None
        if "damaged" in damage.lower():
            dmg_annotated = result_dmg[0].plot()
            dmg_annotated = cv2.cvtColor(dmg_annotated, cv2.COLOR_BGR2RGB)
            buf2 = io.BytesIO()
            Image.fromarray(dmg_annotated).save(buf2, format="PNG")
            damage_img_bytes = buf2.getvalue()

        with dl_cols[0]:
            if dirty_img_bytes:
                st.download_button(
                    "⬇️ Скачать 'Dirty' слой (PNG)",
                    data=dirty_img_bytes,
                    file_name="dirty_annotated.png",
                    mime="image/png"
                )
            else:
                st.caption("Нет 'Dirty' слоя для скачивания.")

        with dl_cols[1]:
            if damage_img_bytes:
                st.download_button(
                    "⬇️ Скачать 'Damage' слой (PNG)",
                    data=damage_img_bytes,
                    file_name="damage_annotated.png",
                    mime="image/png"
                )
            else:
                st.caption("Нет 'Damage' слоя для скачивания.")

    with tabs[2]:
        st.markdown("#### Подсказки по UX")
        st.write(
            "- Загружай фото без сильных бликов и с нормальным освещением.\n"
            "- Снимай под углом ¾, чтобы модель видела больше поверхности.\n"
            "- Если ничего не детектится, попробуй другое фото или кадр покрупнее."
        )

    