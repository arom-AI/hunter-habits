import streamlit as st
import pandas as pd
import datetime
from datetime import date, timedelta
from pathlib import Path
import base64
import json
import re
import gspread
from gspread_dataframe import get_as_dataframe, set_with_dataframe
from openai import OpenAI

# ---------------------------------------------------------------------
# PATHS & DIRECTORIES
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
ASSETS_DIR = BASE_DIR / "assets"

# ---------------------------------------------------------------------
# OPENAI / GPT SYSTEM FUNCTION
# ---------------------------------------------------------------------

def ask_system(prompt: str) -> str:
    """
    Appelle le modèle GPT et génère une réponse stylée ‘SYSTEM’.
    Compatible avec la nouvelle API OpenAI.
    """
    client = OpenAI(api_key=st.secrets["openai"]["api_key"])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu es le SYSTEM du Hunter Discipline Game. "
                    "Tu parles comme dans Solo Leveling : concis, analytique, puissant. "
                    "Tu analyses la discipline, la psychologie, les routines, le mental. "
                    "Tu fournis des quêtes, des mini-objectifs, des diagnostics. "
                    "Tu détectes les chutes, les faiblesses, les patterns. "
                    "Tu encourages, mais toujours avec précision chirurgicale."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=350,
    )

    # Nouveau format : message est un objet, pas un dict
    return response.choices[0].message.content


# ---------------------------------------------------------------------
# AVATAR SELECTION & UTILS
# ---------------------------------------------------------------------

def get_avatar_path_for_level(level: int) -> Path:
    if level <= 10:
        return ASSETS_DIR / "avatar_lvl_1_10.png"
    elif level <= 30:
        return ASSETS_DIR / "avatar_lvl_11_30.png"
    elif level <= 50:
        return ASSETS_DIR / "avatar_lvl_31_50.png"
    else:
        return ASSETS_DIR / "avatar_lvl_50_plus.png"


def image_to_base64(path: Path) -> str:
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_aura_class_for_score(score: int) -> str:
    if score >= 90:
        return "aura-max"
    elif score >= 70:
        return "aura-high"
    elif score >= 40:
        return "aura-mid"
    elif score > 0:
        return "aura-low"
    return "aura-none"

# ---------------------------------------------------------------------
# SOLO LEVELING UI THEME (CSS)
# ---------------------------------------------------------------------

st.markdown("""
<style>

body {
    background-color: #000000 !important;
    color: #e5e7eb !important;
    font-family: 'Inter', sans-serif;
}

/* GLOBAL TEXT */
h1, h2, h3, h4, h5, h6 {
    color: #f3f4f6 !important;
    font-weight: 700;
}

/* SCROLLBAR */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-thumb {
    background: #4b5563;
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
    background: #6b7280;
}

/* SYSTEM MESSAGE */
.system-message {
    margin-top: 15px;
    padding: 16px;
    background: rgba(17, 24, 39, 0.75);
    border: 1px solid rgba(147, 197, 253, 0.35);
    box-shadow: 0 0 16px rgba(96, 165, 250, 0.25);
    border-radius: 12px;
    font-size: 0.95rem;
    color: #e0e7ff;
    backdrop-filter: blur(6px);
}

/* HUNTER CARD */
.hunter-card {
    padding: 18px;
    background: linear-gradient(145deg, #0f172a, #1e293b);
    border-radius: 16px;
    border: 1px solid rgba(148,163,184,0.18);
    box-shadow: 0 0 20px rgba(30,58,138,0.25);
}

.hunter-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #93c5fd;
    margin-bottom: 10px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

.hunter-level {
    font-size: 2rem;
    font-weight: 800;
    color: #fef08a;
    text-shadow: 0 0 8px rgba(250,204,21,0.6);
    margin-bottom: 6px;
}

.hunter-sub {
    color: #9ca3af;
    font-size: 0.9rem;
    margin-top: 4px;
}

/* AVATAR BOX */
.avatar-wrapper {
    width: 140px;
    height: 140px;
    border-radius: 18px;
    overflow: hidden;
    image-rendering: pixelated;
    background: radial-gradient(circle at center, #000212 0%, #000000 80%);
    display: flex;
    align-items: center;
    justify-content: center;
    border: 2px solid rgba(59,130,246,0.6);
    margin-bottom: 14px;
    transition: 0.25s ease-in-out;
}
.avatar-img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    image-rendering: pixelated;
}

/* AURA: pixel-glow */
.aura-none { box-shadow: none; }

.aura-low {
    box-shadow:
        0 0 4px rgba(37, 99, 235, 0.4),
        0 0 10px rgba(59, 130, 246, 0.3);
}

.aura-mid {
    box-shadow:
        0 0 6px rgba(79,70,229,0.6),
        0 0 14px rgba(129,140,248,0.6),
        0 0 20px rgba(59,130,246,0.6);
}

.aura-high {
    box-shadow:
        0 0 10px rgba(124,58,237,1),
        0 0 22px rgba(99,102,241,1),
        0 0 30px rgba(56,189,248,1);
}

.aura-max {
    box-shadow:
        0 0 14px rgba(168,85,247,1),
        0 0 26px rgba(129,140,248,1),
        0 0 36px rgba(96,165,250,1),
        0 0 48px rgba(59,130,246,1);
    animation: auraPulse 1.2s infinite alternate ease-in-out;
}

@keyframes auraPulse {
    0% { transform: translateY(0px) scale(1); }
    100% { transform: translateY(-3px) scale(1.04); }
}

/* TAB THEME */
.stTabs [role="tablist"] {
    border-bottom: 1px solid rgba(148,163,184,0.28);
}
.stTabs [role="tab"] {
    font-weight: 600;
    color: #d1d5db;
}
.stTabs [role="tab"][aria-selected="true"] {
    color: #93c5fd;
    border-bottom-color: #93c5fd !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# GOOGLE SHEETS - SETUP
# ---------------------------------------------------------------------

SHEET_ID = st.secrets["google_sheets"]["sheet_id"]

if "gcp_service_account" in st.secrets:
    gc = gspread.service_account_from_dict(st.secrets["gcp_service_account"])
else:
    SERVICE_ACCOUNT_FILE = BASE_DIR / "habit-builder-streamlit-838d1b385e36.json"
    gc = gspread.service_account(filename=str(SERVICE_ACCOUNT_FILE))

sh = gc.open_by_key(SHEET_ID)

ws_habits = sh.worksheet("habits")
ws_checkins = sh.worksheet("checkins")
ws_meta = sh.worksheet("meta")



# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------

# On charge la feuille 'habits' avec tes colonnes actuelles :
# id | name | category | unit | daily_target | start_date | is_core
habits_df = get_as_dataframe(ws_habits, evaluate_formulas=True, header=0)

# On garde uniquement les lignes avec un nom d'habitude
habits_df = habits_df.dropna(subset=["name"])

# On renomme 'name' en 'habit' pour le reste du code
habits_df = habits_df.rename(columns={"name": "habit"})

# Normalisation de la fréquence : daily / weekly (par défaut : daily)
if "frequency" not in habits_df.columns:
    habits_df["frequency"] = "daily"
else:
    habits_df["frequency"] = (
        habits_df["frequency"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"quotidienne": "daily", "hebdo": "weekly", "hebdomadaire": "weekly"})
    )

# Si la colonne xp_value n'existe pas, on la dérive de is_core :
#  - 20 XP si is_core == 1
#  - 10 XP sinon
if "xp_value" not in habits_df.columns:
    if "is_core" in habits_df.columns:
        def compute_xp(row):
            try:
                return 20 if int(row.get("is_core", 0)) == 1 else 10
            except Exception:
                return 10
        habits_df["xp_value"] = habits_df.apply(compute_xp, axis=1)
    else:
        habits_df["xp_value"] = 10



checkins_df = get_as_dataframe(ws_checkins, evaluate_formulas=True, header=0)
checkins_df = checkins_df.dropna(subset=["date"]) if "date" in checkins_df.columns else pd.DataFrame(columns=["date", "habit_id", "value"])

meta_df = get_as_dataframe(ws_meta, evaluate_formulas=True, header=0)
if "xp_total" in meta_df.columns and not meta_df.empty:
    xp_total = int(meta_df.iloc[0]["xp_total"])
else:
    xp_total = 0

# ---------------------------------------------------------------------
# XP / LEVEL SYSTEM
# ---------------------------------------------------------------------

XP_PER_LEVEL = 100

def get_level(xp: int) -> int:
    return xp // XP_PER_LEVEL + 1

def xp_in_current_level(xp: int) -> int:
    return xp % XP_PER_LEVEL

level = get_level(xp_total)
xp_in_level = xp_in_current_level(xp_total)
progress_level = xp_in_level / XP_PER_LEVEL

# ---------------------------------------------------------------------
# TRACKING TODAY'S DATA
# ---------------------------------------------------------------------

today = date.today()
today_str = today.strftime("%Y-%m-%d")

today_checkins = checkins_df[checkins_df["date"] == today_str] if "date" in checkins_df.columns else pd.DataFrame()

# ---------------------------------------------------------------------
# DAILY DISCIPLINE SCORE
# ---------------------------------------------------------------------

def compute_discipline_score(df):
    """
    Calcule un score de discipline entre 0-100 basé
    sur les habitudes complétées, pondérées par importance.
    """
    if df.empty:
        return 0

    total_weight = df["xp"].sum()
    achieved = df[df["value"] == 1]["xp"].sum()

    if total_weight == 0:
        return 0

    return int((achieved / total_weight) * 100)

def get_today_discipline(checkins_day):
    """
    Calcule le score de discipline du jour
    uniquement sur les habitudes DAILY.
    Les weekly ne pénalisent pas le score.
    """
    if checkins_day.empty:
        return 0

    merged = checkins_day.merge(habits_df, left_on="habit_id", right_on="id", how="left")
    merged = merged[merged["frequency"] == "daily"]

    if merged.empty:
        return 0

    merged = merged.rename(columns={"xp_value": "xp"})
    return compute_discipline_score(merged)


today_score = get_today_discipline(today_checkins)

# ---------------------------------------------------------------------
# DAILY SCORE HISTORY
# ---------------------------------------------------------------------

def build_daily_score_history():
    if checkins_df.empty:
        return pd.DataFrame(columns=["date", "discipline_score"])

    merged = checkins_df.merge(habits_df, left_on="habit_id", right_on="id", how="left")
    merged = merged[merged["frequency"] == "daily"]  # on ne garde que les daily
    merged = merged.rename(columns={"xp_value": "xp"})

    if merged.empty:
        return pd.DataFrame(columns=["date", "discipline_score"])

    history = (
        merged.groupby("date")
        .apply(lambda day: compute_discipline_score(day))
        .reset_index(name="discipline_score")
    )
    return history


daily_scores_df = build_daily_score_history()

def get_week_bounds(d: date):
    """
    Retourne le lundi et le dimanche de la semaine de la date d.
    """
    start = d - timedelta(days=d.weekday())  # lundi
    end = start + timedelta(days=6)          # dimanche
    return start, end


def system_assign_weekly_xp(habit_row, completions_count_last_4_weeks: int) -> int:
    """
    Demande au SYSTEM combien d'XP donner pour une weekly,
    en fonction de l'habitude et de ton historique.
    Si ça foire, fallback sur une valeur raisonnable.
    """
    base_xp = int(habit_row.get("xp_value", 20))
    habit_name = str(habit_row.get("habit", "Habitude"))

    prompt = f"""
    Tu es le SYSTEM d'un jeu de discipline Solo Leveling.

    On vient de valider une HABITUDE HEBDOMADAIRE.

    - Nom de l'habitude : {habit_name}
    - XP de base prévue : {base_xp}
    - Nombre de fois complétée sur les 4 dernières semaines : {completions_count_last_4_weeks}

    Donne UNIQUEMENT un nombre entier d'XP à accorder pour cette complétion.
    Plus c'est rare / difficile pour l'utilisateur, plus l'XP peut être élevée.
    Ne réponds que par le nombre, sans texte autour.
    """

    try:
        raw = ask_system(prompt)
        m = re.search(r"\d+", raw)
        if m:
            xp = int(m.group(0))
            # sécurité : bornes min / max
            return max(10, min(xp, 200))
    except Exception:
        pass

    # fallback
    if completions_count_last_4_weeks == 0:
        return base_xp * 2
    return base_xp


# ---------------------------------------------------------------------
# SAVE CHECKINS / XP / META
# ---------------------------------------------------------------------

def save_checkin(habit_id: int, value: int):
    global checkins_df

    new_row = {
        "date": today_str,
        "habit_id": habit_id,
        "value": value
    }
    checkins_df = pd.concat([checkins_df, pd.DataFrame([new_row])], ignore_index=True)
    set_with_dataframe(ws_checkins, checkins_df)

def save_meta():
    """
    Sauvegarde l'XP totale et autres infos futures.
    """
    meta_save = pd.DataFrame([{"xp_total": xp_total}])
    set_with_dataframe(ws_meta, meta_save)

# ---------------------------------------------------------------------
# SYSTEM GPT — INTELLIGENCE AUTONOME (Version C)
# ---------------------------------------------------------------------

def system_generate_feedback(daily_scores_df, habits_df, xp_total, level):
    """
    Feedback quotidien intelligent.
    Le System analyse ton niveau, ton XP, ton historique de discipline.
    """
    history_json = daily_scores_df.tail(14).to_json()

    prompt = f"""
    Analyse les 14 derniers jours de discipline (0 à 100) :
    {history_json}

    Niveau actuel : {level}
    XP totale : {xp_total}

    Donne un feedback SOLO LEVELING :
    - concis
    - stylé
    - percutant
    - analyse psychologique
    - 2 axes d'amélioration
    - 1 mini-quest du jour (simple, réaliste)
    """

    return ask_system(prompt)


def system_generate_monthly_calibration(daily_scores_df, level, xp_total):
    """
    Calibration mensuelle : 
    une analyse approfondie et la définition des quêtes du mois.
    """
    hist_json = daily_scores_df.to_json()

    prompt = f"""
    Tu es le SYSTEM. L'utilisateur commence un NOUVEAU MOIS.
    Voici tout son historique de discipline :
    {hist_json}

    Niveau actuel : {level}
    XP totale : {xp_total}

    Fais une calibration Solo Leveling :
    - Diagnostic général (forces/faiblesses)
    - 3 objectifs majeurs (Main Quests)
    - 3 objectifs mineurs (Shadow Quests)
    - 1 transformation personnelle à viser ce mois
    """

    return ask_system(prompt)


def system_generate_drop_alert(daily_scores_df):
    """
    Si ton score baisse fortement, le System déclenche une alerte.
    """
    if len(daily_scores_df) < 10:
        return None

    last5 = daily_scores_df.tail(5)["discipline_score"].mean()
    prev5 = daily_scores_df.tail(10).head(5)["discipline_score"].mean()

    if last5 < prev5 - 15:
        prompt = f"""
        L'utilisateur montre une baisse de discipline :
        Moyenne précédente : {prev5}
        Moyenne récente : {last5}

        Donne une ALERT SOLO LEVELING :
        - Ton diagnostic
        - Ce qui cause probablement cette chute
        - Une mini-quest urgente
        - Un message puissant
        """
        return ask_system(prompt)

    return None


def system_generate_level_up_message(level):
    """
    Message SOLO LEVELING quand un niveau est atteint.
    """
    prompt = f"""
    L'utilisateur vient d'atteindre le niveau {level}.
    Génère un message 'LEVEL UP' façon Solo Leveling :
    - puissant
    - court
    - immersif
    - style interface du SYSTEM
    """
    return ask_system(prompt)


# ---------------------------------------------------------------------
# TRIGGERS SYSTEM
# ---------------------------------------------------------------------

system_messages = []  # Liste qui stocke les messages générés

# 1. Trigger : calibration mensuelle
if today.day == 1:
    calibration_msg = system_generate_monthly_calibration(daily_scores_df, level, xp_total)
    system_messages.append(("CALIBRATION", calibration_msg))

# 2. Trigger : détection de chute
alert_msg = system_generate_drop_alert(daily_scores_df)
if alert_msg:
    system_messages.append(("ALERTE", alert_msg))

# 3. Trigger : feedback quotidien (si score enregistré)
if today_score > 0:
    daily_feedback = system_generate_feedback(daily_scores_df, habits_df, xp_total, level)
    system_messages.append(("FEEDBACK", daily_feedback))

# ---------------------------------------------------------------------
# UI: HUNTER CARD + AVATAR + AURA + SYSTEM MESSAGES
# ---------------------------------------------------------------------

st.title("🔥 HUNTER DISCIPLINE SYSTEM — Solo Leveling Edition")

col_left, col_mid, col_right = st.columns([1.3, 1.2, 1.0])

# ---------------------------------------------------------
# LEFT: HUNTER PROFILE
# ---------------------------------------------------------
with col_left:
    st.markdown('<div class="hunter-card">', unsafe_allow_html=True)
    st.markdown('<div class="hunter-title">HUNTER PROFILE</div>', unsafe_allow_html=True)

    # Avatar + Aura
    avatar_path = get_avatar_path_for_level(level)
    avatar_b64 = image_to_base64(avatar_path)
    aura_class = get_aura_class_for_score(today_score)

    st.markdown(
        f"""
        <div class="avatar-wrapper {aura_class}">
            <img class="avatar-img" src="data:image/png;base64,{avatar_b64}" />
        </div>
        """,
        unsafe_allow_html=True
    )

    # Level display
    st.markdown(f'<div class="hunter-level">LVL {level}</div>', unsafe_allow_html=True)
    st.progress(progress_level)

    st.markdown(
        f'<div class="hunter-sub">XP Total : {xp_total} · XP actuel : {xp_in_level}/{XP_PER_LEVEL}</div>',
        unsafe_allow_html=True
    )

    if today_score > 0:
        st.markdown(
            f'<div class="hunter-sub" style="margin-top:10px;">Discipline Score (aujourd\'hui) : <b>{today_score}</b>/100</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="hunter-sub" style="margin-top:10px;">Aucun check-in aujourd’hui.</div>',
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# MIDDLE: SYSTEM MESSAGES
# ---------------------------------------------------------
with col_mid:
    st.markdown("### 💠 SYSTEM LOG")

    if len(system_messages) == 0:
        st.markdown(
            '<div class="system-message">Aucun message du SYSTEM pour le moment.</div>',
            unsafe_allow_html=True
        )
    else:
        for mtype, msg in system_messages:
            st.markdown(
                f"""
                <div class="system-message">
                    <b>[{mtype}]</b><br><br>
                    {msg}
                </div>
                """,
                unsafe_allow_html=True
            )

    # Manual Query to SYSTEM
    st.markdown("### 🔹 Interaction")

    manual_query = st.text_input("Pose une question au SYSTEM :")
    if st.button("Demander"):
        if manual_query.strip():
            answer = ask_system(manual_query)
            st.markdown(
                f"""
                <div class="system-message">
                    <b>[INTERACTION]</b><br><br>
                    {answer}
                </div>
                """,
                unsafe_allow_html=True
            )


# ---------------------------------------------------------
# RIGHT: CALIBRATION + MONTHLY QUEST BUTTON
# ---------------------------------------------------------
with col_right:
    st.markdown("### 🛠 Calibration")

    if st.button("Calibration immédiate (manual override)"):
        msg = system_generate_monthly_calibration(daily_scores_df, level, xp_total)
        st.markdown(
            f'<div class="system-message"><b>[CALIBRATION]</b><br><br>{msg}</div>',
            unsafe_allow_html=True
        )

# ---------------------------------------------------------------------
# DAILY CHECK-IN PANEL
# ---------------------------------------------------------------------

st.markdown("## 📆 Daily Check-In")

if habits_df.empty:
    st.warning("⚠️ Aucune habitude définie dans Google Sheets (onglet 'habits').")
else:
    # Séparation des daily / weekly
    daily_habits = habits_df[habits_df["frequency"] == "daily"]
    weekly_habits = habits_df[habits_df["frequency"] == "weekly"]

    # ---------- DAILY ----------
    st.markdown("### 🔁 Habitudes quotidiennes")

    if daily_habits.empty:
        st.info("Aucune habitude quotidienne définie.")
    else:
        st.markdown(
            "Complète tes habitudes ci-dessous pour gagner du XP et renforcer ton aura."
        )

        updated_today = False
        new_xp_gain = 0

        for idx, row in daily_habits.iterrows():
            habit_id = int(row["id"])
            habit_name = row["habit"]
            xp_value = int(row["xp_value"])

            # Vérifie si déjà rempli aujourd’hui (tous types confondus)
            exists = today_checkins[
                (today_checkins["habit_id"] == habit_id)
            ]

            col1, col2 = st.columns([2, 1])

            with col1:
                st.write(f"**{habit_name}**  (+{xp_value} XP)")

            with col2:
                if exists.empty:
                    if st.button(f"Valider {habit_name}", key=f"habit_daily_{habit_id}"):
                        save_checkin(habit_id, 1)
                        new_xp_gain += xp_value
                        updated_today = True
                else:
                    st.success("✔ OK")

        # Si le joueur a validé des habitudes DAILY aujourd'hui
        if updated_today:
            xp_total += new_xp_gain
            save_meta()

            st.success(f"🔥 {new_xp_gain} XP gagné aujourd'hui !")

            # LEVEL UP ?
            new_level = get_level(xp_total)
            if new_level > level:
                level_up_msg = system_generate_level_up_message(new_level)
                st.markdown(
                    f"""
                    <div class="system-message">
                        <b>[LEVEL UP]</b><br><br>
                        {level_up_msg}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.experimental_rerun()

    # ---------- WEEKLY ----------
    st.markdown("---")
    st.markdown("### 📅 Habitudes hebdomadaires (1x par semaine)")

    if weekly_habits.empty:
        st.info("Aucune habitude hebdomadaire définie.")
    else:
        week_start, week_end = get_week_bounds(today)
        week_start_str = week_start.strftime("%Y-%m-%d")
        week_end_str = week_end.strftime("%Y-%m-%d")

        for idx, row in weekly_habits.iterrows():
            habit_id = int(row["id"])
            habit_name = row["habit"]

            # Check : déjà validée cette semaine ?
            mask = (
                (checkins_df["habit_id"] == habit_id) &
                (checkins_df["date"] >= week_start_str) &
                (checkins_df["date"] <= week_end_str)
            )
            done_this_week = not checkins_df[mask].empty

            col1, col2 = st.columns([2, 1])

            with col1:
                st.write(f"**{habit_name}**  (weekly)")

            with col2:
                if done_this_week:
                    st.success("✔ Validée cette semaine")
                else:
                    if st.button(f"Valider {habit_name}", key=f"habit_weekly_{habit_id}"):
                        # Nombre de complétions sur 4 semaines
                        four_weeks_ago = today - timedelta(weeks=4)
                        four_weeks_ago_str = four_weeks_ago.strftime("%Y-%m-%d")
                        mask_hist = (
                            (checkins_df["habit_id"] == habit_id) &
                            (checkins_df["date"] >= four_weeks_ago_str)
                        )
                        completions_last_4_weeks = checkins_df[mask_hist].shape[0]

                        # XP dynamique via SYSTEM
                        gained_xp = system_assign_weekly_xp(row, completions_last_4_weeks)
                        xp_total += gained_xp

                        save_checkin(habit_id, 1)
                        save_meta()

                        st.success(f"💠 Weekly complétée : +{gained_xp} XP")
                        st.experimental_rerun()


# ---------------------------------------------------------------------
# MANAGE HABITS (OPTIONAL AREA)
# ---------------------------------------------------------------------

st.markdown("---")
st.markdown("### ⚙️ Gérer les habitudes")

with st.expander("Ajouter une nouvelle habitude"):
    new_habit_name = st.text_input("Nom de l'habitude :")
    new_habit_xp = st.number_input("XP associée :", min_value=5, max_value=100, value=10, step=5)
    new_habit_freq = st.selectbox("Fréquence", ["daily", "weekly"], index=0)

    if st.button("Ajouter l'habitude"):
        if new_habit_name.strip():
            new_id = habits_df["id"].max() + 1 if not habits_df.empty else 1
            new_row = {
                "id": new_id,
                "habit": new_habit_name,
                "xp_value": new_habit_xp,
                "frequency": new_habit_freq,
            }
            habits_df = pd.concat([habits_df, pd.DataFrame([new_row])], ignore_index=True)
            set_with_dataframe(ws_habits, habits_df)
            st.success("Habitude ajoutée.")
            st.experimental_rerun()



with st.expander("Modifier / supprimer des habitudes"):
    st.write(habits_df[["id", "habit", "xp_value"]])

    mod_id = st.number_input("ID de l'habitude à modifier/supprimer :", min_value=1)
    new_name = st.text_input("Nouveau nom (laisser vide pour ne pas changer) :")
    new_xp_val = st.number_input("Nouvelle valeur XP (0 = ne pas changer) :", min_value=0, max_value=200, value=0)

    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("Modifier"):
            if mod_id in habits_df["id"].values:
                if new_name.strip():
                    habits_df.loc[habits_df["id"] == mod_id, "habit"] = new_name
                if new_xp_val > 0:
                    habits_df.loc[habits_df["id"] == mod_id, "xp_value"] = new_xp_val
                set_with_dataframe(ws_habits, habits_df)
                st.success("Habitude modifiée.")
                st.experimental_rerun()

    with col_b:
        if st.button("Supprimer"):
            habits_df = habits_df[habits_df["id"] != mod_id]
            set_with_dataframe(ws_habits, habits_df)
            st.success("Habitude supprimée.")
            st.experimental_rerun()

# ---------------------------------------------------------------------
# GRAPHES — Evolution de la Discipline
# ---------------------------------------------------------------------

st.markdown("## 📈 Evolution de la Discipline (14 jours)")

if daily_scores_df.empty:
    st.info("Aucune donnée enregistrée pour le moment.")
else:
    last14 = daily_scores_df.tail(14)
    st.line_chart(
        last14.set_index("date")["discipline_score"],
        height=220
    )


# ---------------------------------------------------------------------
# HUNTER HISTORY — Timeline
# ---------------------------------------------------------------------

st.markdown("## 🕒 Hunter History")

if daily_scores_df.empty:
    st.info("Aucun historique pour le moment.")
else:
    for idx, row in daily_scores_df.tail(21).iterrows():
        date_str = row["date"]
        score = row["discipline_score"]

        aura_class = get_aura_class_for_score(score)

        st.markdown(
            f"""
            <div style="
                margin-bottom:10px;
                padding:12px;
                border-radius:12px;
                background:rgba(17,24,39,0.7);
                border:1px solid rgba(148,163,184,0.25);
                box-shadow:0 0 10px rgba(37,99,235,0.2);
            ">
                <b>{date_str}</b><br>
                Discipline : <span style="color:#93c5fd;">{score}/100</span>
                <div class="{aura_class}" style="margin-top:4px;width:90%;height:4px;border-radius:4px;background:rgba(59,130,246,0.25);"></div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ---------------------------------------------------------------------
# QUESTS SYSTEM (GPT-Generated Quests)
# ---------------------------------------------------------------------

st.markdown("## 🎯 Quêtes du Jour & du Mois")

col_q1, col_q2 = st.columns(2)

# DAILY QUEST
with col_q1:
    st.markdown("### 🔥 Daily Quest")

    if st.button("Générer une Daily Quest"):
        prompt = f"""
        Génère UNE seule Daily Quest Solo Leveling.
        Basée sur les habitudes et l'énergie récente.
        Score d'aujourd'hui = {today_score}
        Historique discipline = {daily_scores_df.tail(10).to_dict()}

        Style SYSTEM :
        - simple mais impactante
        - faisable aujourd'hui
        - orientée progression
        """
        q = ask_system(prompt)
        st.markdown(f'<div class="system-message">{q}</div>', unsafe_allow_html=True)


# MONTHLY QUEST
with col_q2:
    st.markdown("### 💎 Monthly Quest")

    if st.button("Générer une Monthly Quest"):
        prompt = f"""
        Génère une Monthly Quest complète :
        - 1 Main Quest
        - 2 Shadow Quests
        - 1 Transformation personnelle
        Niveau = {level}
        XP totale = {xp_total}
        Historique = {daily_scores_df.to_dict()}
        Style SYSTEM.
        """
        q = ask_system(prompt)
        st.markdown(f'<div class="system-message">{q}</div>', unsafe_allow_html=True)



# ---------------------------------------------------------------------
# OPTIONAL: BACKGROUND IMAGE SUPPORT
# (Use uploaded screenshot as background if wanted)
# ---------------------------------------------------------------------

background_image_path = "/mnt/data/Capture d’écran 2025-11-18 à 20.30.17.png"

def inject_background_image(path):
    try:
        import base64
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background: url("data:image/png;base64,{encoded}") !important;
                background-size: cover !important;
                background-position: center !important;
                background-attachment: fixed !important;
                opacity: 1;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except:
        pass

# Toggle
st.markdown("## 🎨 Apparence (Optionnel)")

if st.toggle("Activer fond personnalisé"):
    inject_background_image(background_image_path)


# ---------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------

st.markdown("""
<hr>
<div style='text-align:center;color:#475569;font-size:0.85rem;padding:10px;'>
    HUNTER SYSTEM • Solo Leveling Edition<br>
    Powered by Streamlit + GPT + Discipline<br>
    <span style='font-size:0.7rem;'>“Arise.”</span>
</div>
""", unsafe_allow_html=True)
