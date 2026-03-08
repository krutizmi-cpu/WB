
import json
import math
import re
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
TEMPLATES_DIR = ROOT / "templates"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="WB Unit Economics", page_icon="📦", layout="wide")

@st.cache_data
def load_config():
    with open(ROOT / "config.json", "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_patterns():
    with open(DATA_DIR / "category_patterns.json", "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_commissions():
    df = pd.read_excel(DATA_DIR / "wb_commissions.xlsx")
    return df.fillna("")

def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = text.replace("ё", "е")
    text = re.sub(r"[^a-zа-я0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def detect_category(name: str, patterns: list[dict]) -> tuple[str, str]:
    text = normalize_text(name)
    best_category = "Прочее"
    best_pattern = ""
    best_score = -1
    for item in patterns:
        for pattern in item["patterns"]:
            p = normalize_text(pattern)
            if p in text:
                score = len(p)
                if score > best_score:
                    best_score = score
                    best_category = item["category"]
                    best_pattern = pattern
    return best_category, best_pattern

def get_commission_pct(category: str, model: str, commission_df: pd.DataFrame) -> float:
    row = commission_df.loc[commission_df["Категория"].astype(str).str.lower() == category.lower()]
    if row.empty:
        row = commission_df.loc[commission_df["Категория"].astype(str).str.lower() == "прочее"]
    if row.empty:
        return 18.0
    return float(row.iloc[0]["Комиссия FBW/FBO, %"] if model == "FBW" else row.iloc[0]["Комиссия FBS, %"])

def volume_liters(length_cm: float, width_cm: float, height_cm: float) -> float:
    return max(length_cm, 0) * max(width_cm, 0) * max(height_cm, 0) / 1000.0

def wb_fbw_logistics_rub(volume_l: float) -> float:
    """Правило по официальной статье WB на 15.09.2025:
    до 1 литра включительно зависит от диапазона;
    свыше 1 литра: 46 ₽ за первый литр + 14 ₽ за каждый дополнительный литр.
    """
    if volume_l <= 0:
        return 0.0
    if volume_l <= 0.2:
        return 23.0
    if volume_l <= 0.4:
        return 26.0
    if volume_l <= 0.6:
        return 29.0
    if volume_l <= 0.8:
        return 30.0
    if volume_l <= 1.0:
        return 32.0
    return 46.0 + max(volume_l - 1.0, 0) * 14.0

def safe_pct(value) -> float:
    try:
        return float(value) / 100.0
    except Exception:
        return 0.0

def rub(value) -> float:
    try:
        if pd.isna(value):
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")

def compute_recommended_price(unit_cost_total: float, commission_pct: float, ad_pct: float, acquiring_pct: float,
                              tax_mode_code: str, tax_rate: float, target_margin_pct: float) -> float:
    target_margin = target_margin_pct / 100.0
    comm = commission_pct / 100.0
    ad = ad_pct / 100.0
    acq = acquiring_pct / 100.0

    if tax_mode_code == "USN_PROFIT_15":
        denom = 1 - target_margin - comm - ad - acq
        if denom <= 0:
            return float("nan")
        price_before_tax = unit_cost_total / denom
        profit_before_tax = price_before_tax - unit_cost_total - price_before_tax * (comm + ad + acq)
        tax = max(profit_before_tax, 0) * tax_rate
        final_profit = profit_before_tax - tax
        if price_before_tax <= 0:
            return float("nan")
        if final_profit / price_before_tax < target_margin:
            gross_up = target_margin / max(final_profit / price_before_tax, 1e-9)
            return price_before_tax * gross_up
        return price_before_tax

    denom = 1 - target_margin - comm - ad - acq - tax_rate
    if denom <= 0:
        return float("nan")
    return unit_cost_total / denom

def compute_unit_economics(df: pd.DataFrame, ui: dict, config: dict) -> pd.DataFrame:
    patterns = load_patterns()
    commission_df = load_commissions()
    tax_mode_code = ui["tax_mode"]
    tax_rate = config["tax_modes"][tax_mode_code]["rate"]

    results = []
    for _, row in df.iterrows():
        article = row.get("Артикул", "")
        name = row.get("Наименование", "")
        length_cm = rub(row.get("Длина, см", 0))
        width_cm = rub(row.get("Ширина, см", 0))
        height_cm = rub(row.get("Высота, см", 0))
        weight_kg = rub(row.get("Вес, кг (опц.)", float("nan")))
        cogs = rub(row.get("Себестоимость, ₽ (опц.)", float("nan")))
        seller_price = rub(row.get("Цена продавца до СПП, ₽ (опц.)", float("nan")))
        sales_model_raw = str(row.get("Модель продаж (опц.: FBS/FBW)", "")).strip().upper()
        fbw_days_row = rub(row.get("Дней хранения FBW/FBO (опц.)", float("nan")))

        if math.isnan(cogs):
            cogs = ui["default_cogs"]
        if math.isnan(seller_price):
            seller_price = ui["default_price"]

        model = sales_model_raw if sales_model_raw in {"FBS", "FBW", "FBO"} else ui["default_model"]
        if model == "FBO":
            model = "FBW"

        category, matched_pattern = detect_category(name, patterns)
        commission_pct = get_commission_pct(category, model, commission_df)

        volume_l = volume_liters(length_cm, width_cm, height_cm)
        base_logistics = wb_fbw_logistics_rub(volume_l)
        if model == "FBS":
            outbound_logistics = base_logistics * ui["fbs_logistics_coef"]
            storage = 0.0
        else:
            outbound_logistics = base_logistics
            storage_days = int(fbw_days_row) if not math.isnan(fbw_days_row) and fbw_days_row >= 0 else int(ui["fbw_storage_days"])
            storage = volume_l * ui["fbw_storage_rub_per_liter_day"] * storage_days

        buyout_rate = max(min(ui["buyout_pct"] / 100.0, 0.999), 0.01)
        non_buyout_rate = 1 - buyout_rate
        seller_discount_rate = ui["seller_discount_pct"] / 100.0
        spp_rate = ui["spp_pct"] / 100.0

        seller_discounted_price = seller_price * (1 - seller_discount_rate)
        buyer_price_after_spp = seller_discounted_price * (1 - spp_rate)

        shipped_per_successful_sale = 1 / buyout_rate
        effective_outbound = outbound_logistics * shipped_per_successful_sale
        effective_reverse = outbound_logistics * ui["reverse_logistics_coef"] * non_buyout_rate * shipped_per_successful_sale
        effective_storage = storage * shipped_per_successful_sale if model == "FBW" else storage

        commission_rub = seller_discounted_price * (commission_pct / 100.0)
        ad_rub = seller_discounted_price * (ui["ad_pct"] / 100.0)
        acquiring_rub = seller_discounted_price * (ui["acquiring_pct"] / 100.0)
        defect_rub = cogs * (ui["defect_pct"] / 100.0) * shipped_per_successful_sale

        pre_tax_profit = seller_discounted_price - cogs - commission_rub - ad_rub - acquiring_rub - effective_outbound - effective_reverse - effective_storage - defect_rub

        if tax_mode_code == "USN_PROFIT_15":
            tax_rub = max(pre_tax_profit, 0) * tax_rate
        else:
            tax_rub = seller_discounted_price * tax_rate

        net_profit = pre_tax_profit - tax_rub
        margin_pct = (net_profit / seller_discounted_price * 100.0) if seller_discounted_price else 0.0

        full_cost = cogs + commission_rub + ad_rub + acquiring_rub + effective_outbound + effective_reverse + effective_storage + defect_rub + tax_rub
        markup_pct = (seller_discounted_price / full_cost - 1) * 100.0 if full_cost > 0 else 0.0

        recommended_price = compute_recommended_price(
            unit_cost_total=(cogs + effective_outbound + effective_reverse + effective_storage + defect_rub),
            commission_pct=commission_pct,
            ad_pct=ui["ad_pct"],
            acquiring_pct=ui["acquiring_pct"],
            tax_mode_code=tax_mode_code,
            tax_rate=tax_rate,
            target_margin_pct=ui["target_margin_pct"],
        )
        recommended_buyer_price = recommended_price * (1 - seller_discount_rate) * (1 - spp_rate) if pd.notna(recommended_price) else float("nan")

        results.append({
            "Артикул": article,
            "Наименование": name,
            "Категория WB": category,
            "Совпадение по шаблону": matched_pattern,
            "Модель продаж": model,
            "Длина, см": length_cm,
            "Ширина, см": width_cm,
            "Высота, см": height_cm,
            "Вес, кг": None if math.isnan(weight_kg) else weight_kg,
            "Объём, л": round(volume_l, 3),
            "Себестоимость, ₽": round(cogs, 2),
            "Цена продавца до СПП, ₽": round(seller_price, 2),
            "Цена после скидки продавца, ₽": round(seller_discounted_price, 2),
            "Цена покупателя после СПП, ₽": round(buyer_price_after_spp, 2),
            "Комиссия, %": round(commission_pct, 2),
            "Комиссия, ₽": round(commission_rub, 2),
            "Логистика прямая на успешную продажу, ₽": round(effective_outbound, 2),
            "Обратная логистика на успешную продажу, ₽": round(effective_reverse, 2),
            "Хранение, ₽": round(effective_storage, 2),
            "Реклама, ₽": round(ad_rub, 2),
            "Эквайринг, ₽": round(acquiring_rub, 2),
            "Брак/потери, ₽": round(defect_rub, 2),
            "Налог, ₽": round(tax_rub, 2),
            "Полная себестоимость, ₽": round(full_cost, 2),
            "Прибыль, ₽": round(net_profit, 2),
            "Маржа, %": round(margin_pct, 2),
            "Наценка, %": round(markup_pct, 2),
            "Рекомендованная цена продавца до СПП, ₽": round(recommended_price, 2) if pd.notna(recommended_price) else None,
            "Рекомендованная цена покупателя после СПП, ₽": round(recommended_buyer_price, 2) if pd.notna(recommended_buyer_price) else None,
            "Выкупаемость, %": ui["buyout_pct"],
            "СПП, %": ui["spp_pct"],
            "Скидка продавца, %": ui["seller_discount_pct"],
            "Реклама, %": ui["ad_pct"],
            "Брак, %": ui["defect_pct"],
            "Эквайринг, %": ui["acquiring_pct"],
            "Налоговый режим": config["tax_modes"][tax_mode_code]["label"],
        })

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        ordered_cols = [
            "Артикул","Наименование","Категория WB","Совпадение по шаблону","Модель продаж","Объём, л",
            "Себестоимость, ₽","Цена продавца до СПП, ₽","Цена после скидки продавца, ₽","Цена покупателя после СПП, ₽",
            "Комиссия, %","Комиссия, ₽","Логистика прямая на успешную продажу, ₽","Обратная логистика на успешную продажу, ₽",
            "Хранение, ₽","Реклама, ₽","Эквайринг, ₽","Брак/потери, ₽","Налог, ₽","Полная себестоимость, ₽",
            "Прибыль, ₽","Маржа, %","Наценка, %","Рекомендованная цена продавца до СПП, ₽",
            "Рекомендованная цена покупателя после СПП, ₽","Выкупаемость, %","СПП, %","Скидка продавца, %",
            "Реклама, %","Брак, %","Эквайринг, %","Налоговый режим","Длина, см","Ширина, см","Высота, см","Вес, кг"
        ]
        result_df = result_df[ordered_cols]
    return result_df

def to_excel_bytes(df: pd.DataFrame) -> bytes:
    from io import BytesIO
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="WB unit economics")
        ws = writer.book["WB unit economics"]
        for cell in ws[1]:
            cell.font = cell.font.copy(bold=True, color="FFFFFF")
            cell.fill = cell.fill.copy(fill_type="solid", fgColor="1F4E78")
        money_cols = [i+1 for i, c in enumerate(df.columns) if "₽" in c]
        pct_cols = [i+1 for i, c in enumerate(df.columns) if ", %" in c or c.endswith("%")]
        for col_idx in money_cols:
            for r in range(2, ws.max_row+1):
                ws.cell(r, col_idx).number_format = '#,##0.00'
        for col_idx in pct_cols:
            for r in range(2, ws.max_row+1):
                val = ws.cell(r, col_idx).value
                if isinstance(val, (int, float)):
                    ws.cell(r, col_idx).number_format = '0.00'
        for idx, col_name in enumerate(df.columns, start=1):
            width = min(max(len(str(col_name)) + 2, 14), 34)
            ws.column_dimensions[chr(64+idx) if idx <= 26 else "A"].width = width
        ws.freeze_panes = "A2"
    return output.getvalue()

cfg = load_config()
defaults = cfg["defaults"]

st.title("WB unit-экономика — FBS + FBW/FBO")
st.caption("Массовая загрузка Excel, автоопределение категории WB по названию, мгновенный пересчёт при изменении параметров.")

top1, top2 = st.columns([1, 1])
with top1:
    with open(TEMPLATES_DIR / "wb_template.xlsx", "rb") as f:
        st.download_button("Скачать шаблон Excel", data=f.read(), file_name="wb_template.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
with top2:
    with open(DATA_DIR / "wb_commissions.xlsx", "rb") as f:
        st.download_button("Скачать локальный справочник комиссий", data=f.read(), file_name="wb_commissions.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

st.markdown("### Параметры расчёта")
c1, c2, c3, c4 = st.columns(4)
with c1:
    default_model = st.selectbox("Модель продаж по умолчанию", ["FBS", "FBW"], index=0 if cfg["default_model"] == "FBS" else 1)
    tax_mode = st.selectbox("Система налогообложения", list(cfg["tax_modes"].keys()), format_func=lambda x: cfg["tax_modes"][x]["label"])
    target_margin_pct = st.number_input("Целевая маржа, %", min_value=0.0, max_value=95.0, value=float(defaults["target_margin_pct"]), step=1.0)
with c2:
    spp_pct = st.number_input("СПП, %", min_value=0.0, max_value=95.0, value=float(defaults["spp_pct"]), step=0.5)
    seller_discount_pct = st.number_input("Скидка продавца, %", min_value=0.0, max_value=95.0, value=float(defaults["seller_discount_pct"]), step=0.5)
    buyout_pct = st.number_input("Выкупаемость, %", min_value=1.0, max_value=99.9, value=float(defaults["buyout_pct"]), step=0.5)
with c3:
    ad_pct = st.number_input("Реклама, %", min_value=0.0, max_value=95.0, value=float(defaults["ad_pct"]), step=0.5)
    defect_pct = st.number_input("Брак / потери, %", min_value=0.0, max_value=95.0, value=float(defaults["defect_pct"]), step=0.5)
    acquiring_pct = st.number_input("Эквайринг, %", min_value=0.0, max_value=15.0, value=float(defaults["acquiring_pct"]), step=0.1)
with c4:
    reverse_logistics_coef = st.number_input("Коэф. обратной логистики", min_value=0.0, max_value=5.0, value=float(defaults["reverse_logistics_coef"]), step=0.1)
    fbw_storage_days = st.number_input("План хранения FBW/FBO, дней", min_value=0, max_value=3650, value=int(defaults["fbw_storage_days"]), step=1)
    fbw_storage_rub_per_liter_day = st.number_input("Хранение FBW/FBO, ₽/л/день", min_value=0.0, max_value=100.0, value=float(defaults["fbw_storage_rub_per_liter_day"]), step=0.01)

with st.expander("Допущения по умолчанию, если в Excel нет себестоимости/цены"):
    d1, d2, d3 = st.columns(3)
    with d1:
        default_price = st.number_input("Цена продавца до СПП по умолчанию, ₽", min_value=0.0, value=float(defaults["seller_price_before_spp_rub"]), step=10.0)
    with d2:
        default_cogs = st.number_input("Себестоимость по умолчанию, ₽", min_value=0.0, value=float(defaults["cogs_rub"]), step=10.0)
    with d3:
        fbs_logistics_coef = st.number_input("Коэф. логистики FBS к базовому тарифу WB", min_value=0.1, max_value=5.0, value=float(defaults["fbs_logistics_coef"]), step=0.1)

uploaded = st.file_uploader("Загрузите Excel с товарами", type=["xlsx"])

required_columns = ["Артикул", "Наименование", "Длина, см", "Ширина, см", "Высота, см"]
if uploaded:
    raw_df = pd.read_excel(uploaded)
    missing = [c for c in required_columns if c not in raw_df.columns]
    if missing:
        st.error(f"В файле не хватает колонок: {', '.join(missing)}")
        st.stop()

    ui = {
        "default_model": default_model,
        "tax_mode": tax_mode,
        "target_margin_pct": target_margin_pct,
        "spp_pct": spp_pct,
        "seller_discount_pct": seller_discount_pct,
        "buyout_pct": buyout_pct,
        "ad_pct": ad_pct,
        "defect_pct": defect_pct,
        "acquiring_pct": acquiring_pct,
        "reverse_logistics_coef": reverse_logistics_coef,
        "fbw_storage_days": fbw_storage_days,
        "fbw_storage_rub_per_liter_day": fbw_storage_rub_per_liter_day,
        "default_price": default_price,
        "default_cogs": default_cogs,
        "fbs_logistics_coef": fbs_logistics_coef,
    }

    result_df = compute_unit_economics(raw_df, ui, cfg)

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("SKU", len(result_df))
    with s2:
        st.metric("Средняя маржа, %", f"{result_df['Маржа, %'].mean():.2f}")
    with s3:
        st.metric("Средняя прибыль, ₽", f"{result_df['Прибыль, ₽'].mean():,.2f}".replace(",", " "))
    with s4:
        st.metric("Минусовых SKU", int((result_df["Прибыль, ₽"] < 0).sum()))

    st.dataframe(result_df, use_container_width=True, height=650)

    excel_bytes = to_excel_bytes(result_df)
    st.download_button(
        "Скачать результат Excel",
        data=excel_bytes,
        file_name="wb_unit_economics_result.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

    st.markdown("### Что считается сейчас")
    st.info(
        "Пересчёт идёт мгновенно при любом изменении параметров. "
        "Обратная логистика зависит от выкупаемости. "
        "Для FBW/FBO учитывается плановый срок хранения, для FBS хранение = 0."
    )
else:
    st.warning("Сначала скачайте шаблон, заполните его и загрузите обратно.")
