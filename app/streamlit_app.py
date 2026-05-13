import streamlit as st
import pandas as pd
import joblib
import sys
import os

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_PATH)

from src.simulation.tournament import simulate_tournament, simulate_monte_carlo

MODEL_PATH = os.path.join(ROOT_PATH, "trained_models", "modelo_pipeline.pkl")
model = joblib.load(MODEL_PATH)

groups = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["USA", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curacao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"]
}

# MAPEO NOMBRES (display --> dataset)
team_mapping = {
    "USA":          "United States",
    "Curacao":      "Curaçao",
}

# Mapeo inverso para mostrar nombres bonitos
reverse_mapping = {v: k for k, v in team_mapping.items()}

def map_team(team):
    return team_mapping.get(team, team)

def display_name(team):
    return reverse_mapping.get(team, team)


# ELOs REALES (FIFA)

REAL_ELOS = {
    "Mexico":                   1775,
    "South Africa":             1626,
    "South Korea":              1774,
    "Czech Republic":           1661,
    "Canada":                   1733,
    "Bosnia and Herzegovina":   1551,
    "Qatar":                    1603,
    "Switzerland":              1746,
    "Brazil":                   1867,
    "Morocco":                  1841,
    "Haiti":                    1583,
    "Scotland":                 1642,
    "United States":            1820,   
    "Paraguay":                 1687,
    "Australia":                1761,
    "Turkey":                   1736,
    "Germany":                  1831,
    "Curaçao":                  1450,   
    "Ivory Coast":              1710,
    "Ecuador":                  1781,
    "Netherlands":              1831,
    "Japan":                    1839,
    "Sweden":                   1676,
    "Tunisia":                  1682,
    "Belgium":                  1788,
    "Egypt":                    1718,
    "Iran":                     1788,
    "New Zealand":              1603,
    "Spain":                    1938,
    "Cape Verde":               1593,
    "Saudi Arabia":             1632,
    "Uruguay":                  1780,
    "France":                   1906,
    "Senegal":                  1792,
    "Iraq":                     1684,
    "Norway":                   1708,
    "Argentina":                1943,
    "Algeria":                  1768,
    "Austria":                  1726,
    "Jordan":                   1658,
    "Portugal":                 1846,
    "DR Congo":                 1636,
    "Uzbekistan":               1728,
    "Colombia":                 1835,
    "England":                  1847,
    "Croatia":                  1797,
    "Ghana":                    1567,
    "Panama":                   1683,
}

# Forma reciente estimada (puntos promedio por partido últimas 5 fechas FIFA)
FORM_DATA = {
    "Argentina":                {"form_points": 2.4, "goals_for": 2.2, "goals_against": 0.8},
    "Spain":                    {"form_points": 2.3, "goals_for": 2.0, "goals_against": 0.7},
    "France":                   {"form_points": 2.1, "goals_for": 1.8, "goals_against": 0.9},
    "Brazil":                   {"form_points": 2.0, "goals_for": 1.9, "goals_against": 0.9},
    "England":                  {"form_points": 2.0, "goals_for": 1.8, "goals_against": 0.8},
    "Germany":                  {"form_points": 1.9, "goals_for": 1.7, "goals_against": 1.0},
    "Portugal":                 {"form_points": 2.0, "goals_for": 2.0, "goals_against": 0.9},
    "Netherlands":              {"form_points": 1.9, "goals_for": 1.7, "goals_against": 1.0},
    "Morocco":                  {"form_points": 1.9, "goals_for": 1.5, "goals_against": 0.7},
    "Japan":                    {"form_points": 2.0, "goals_for": 1.8, "goals_against": 0.9},
    "Colombia":                 {"form_points": 2.0, "goals_for": 1.8, "goals_against": 0.9},
    "Belgium":                  {"form_points": 1.8, "goals_for": 1.6, "goals_against": 1.0},
    "Croatia":                  {"form_points": 1.8, "goals_for": 1.4, "goals_against": 0.9},
    "Senegal":                  {"form_points": 1.8, "goals_for": 1.5, "goals_against": 0.9},
    "Mexico":                   {"form_points": 1.7, "goals_for": 1.5, "goals_against": 1.1},
    "Ecuador":                  {"form_points": 1.7, "goals_for": 1.5, "goals_against": 1.0},
    "Uruguay":                  {"form_points": 1.7, "goals_for": 1.4, "goals_against": 1.0},
    "United States":            {"form_points": 1.7, "goals_for": 1.5, "goals_against": 1.1},
    "Iran":                     {"form_points": 1.7, "goals_for": 1.5, "goals_against": 1.0},
    "Algeria":                  {"form_points": 1.7, "goals_for": 1.4, "goals_against": 1.0},
    "Turkey":                   {"form_points": 1.6, "goals_for": 1.4, "goals_against": 1.1},
    "South Korea":              {"form_points": 1.6, "goals_for": 1.4, "goals_against": 1.1},
    "Switzerland":              {"form_points": 1.6, "goals_for": 1.3, "goals_against": 1.0},
    "Norway":                   {"form_points": 1.6, "goals_for": 1.6, "goals_against": 1.2},
    "Australia":                {"form_points": 1.6, "goals_for": 1.3, "goals_against": 1.1},
    "Austria":                  {"form_points": 1.6, "goals_for": 1.4, "goals_against": 1.1},
    "Uzbekistan":               {"form_points": 1.6, "goals_for": 1.3, "goals_against": 1.1},
    "Ivory Coast":              {"form_points": 1.5, "goals_for": 1.3, "goals_against": 1.1},
    "Egypt":                    {"form_points": 1.5, "goals_for": 1.2, "goals_against": 1.0},
    "Iraq":                     {"form_points": 1.5, "goals_for": 1.2, "goals_against": 1.1},
    "Paraguay":                 {"form_points": 1.5, "goals_for": 1.2, "goals_against": 1.2},
    "Sweden":                   {"form_points": 1.5, "goals_for": 1.3, "goals_against": 1.1},
    "Scotland":                 {"form_points": 1.5, "goals_for": 1.3, "goals_against": 1.2},
    "Czech Republic":           {"form_points": 1.5, "goals_for": 1.2, "goals_against": 1.1},
    "Canada":                   {"form_points": 1.5, "goals_for": 1.3, "goals_against": 1.2},
    "Tunisia":                  {"form_points": 1.4, "goals_for": 1.1, "goals_against": 1.1},
    "Jordan":                   {"form_points": 1.4, "goals_for": 1.1, "goals_against": 1.2},
    "DR Congo":                 {"form_points": 1.4, "goals_for": 1.1, "goals_against": 1.2},
    "Ghana":                    {"form_points": 1.3, "goals_for": 1.1, "goals_against": 1.3},
    "Panama":                   {"form_points": 1.3, "goals_for": 1.0, "goals_against": 1.2},
    "Saudi Arabia":             {"form_points": 1.3, "goals_for": 1.1, "goals_against": 1.3},
    "New Zealand":              {"form_points": 1.2, "goals_for": 1.0, "goals_against": 1.3},
    "Cape Verde":               {"form_points": 1.3, "goals_for": 1.1, "goals_against": 1.2},
    "South Africa":             {"form_points": 1.2, "goals_for": 1.0, "goals_against": 1.2},
    "Bosnia and Herzegovina":   {"form_points": 1.2, "goals_for": 1.0, "goals_against": 1.3},
    "Qatar":                    {"form_points": 1.1, "goals_for": 0.9, "goals_against": 1.4},
    "Haiti":                    {"form_points": 1.0, "goals_for": 0.9, "goals_against": 1.5},
    "Curaçao":                  {"form_points": 1.0, "goals_for": 0.9, "goals_against": 1.4},
}

def build_features_dict(teams):
    """
    Construye el dict de features con ELOs reales y forma estimada
    para cada equipo del torneo.
    """
    features = {}
    for team in teams:
        elo = REAL_ELOS.get(team, 1500)
        form = FORM_DATA.get(team, {"form_points": 1.3, "goals_for": 1.1, "goals_against": 1.2})
        features[team] = {
            "elo":           elo,
            "form_points":   form["form_points"],
            "goals_for":     form["goals_for"],
            "goals_against": form["goals_against"],
        }
    return features


# UI
st.set_page_config(page_title="Simulador Mundial 2026", layout="wide")

st.title("🌎 Simulador del Mundial 2026")
st.markdown("Simulación completa con un modelo predictivo XGBoost, basado en datos históricos, forma reciente de las selecciones, etc. ¡Descubre quién tiene más chances de levantar la copa! 🏆⚽")


# MOSTRAR GRUPOS

st.subheader("📋 Grupos oficiales")
cols = st.columns(4)

for i, (g, teams) in enumerate(groups.items()):
    with cols[i % 4]:
        st.markdown(f"### Grupo {g}")
        for t in teams:
            st.write(f"- {t}")


# PREPARAR DATOS INTERNOS

mapped_groups = {
    g: [map_team(t) for t in teams]
    for g, teams in groups.items()
}

all_teams = list(set(sum(mapped_groups.values(), [])))
features_ready = build_features_dict(all_teams)


# SIMULACIÓN

if st.button("⚽ Simular Mundial 2026"):


    # MONTE CARLO
    
    with st.spinner("Ejecutando 100 simulaciones Monte Carlo..."):
        probs = simulate_monte_carlo(
            mapped_groups,
            features_ready,
            model,
            n_simulations=100
        )

    df_probs = pd.DataFrame(
        list(probs.items()),
        columns=["Equipo", "Probabilidad (%)"]
    )
    df_probs["Equipo"] = df_probs["Equipo"].map(display_name).fillna(df_probs["Equipo"])
    df_probs["Probabilidad (%)"] = (df_probs["Probabilidad (%)"] * 100).round(2)
    df_probs = df_probs.sort_values(by="Probabilidad (%)", ascending=False).reset_index(drop=True)

    st.subheader("Probabilidad de campeón (Monte Carlo - 100 simulaciones)")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(df_probs, use_container_width=True)
    with col2:
        st.bar_chart(df_probs.set_index("Equipo"))

    st.markdown("---")


    # IMULACIÓN CONCRETA
    
    with st.spinner("Simulando torneo completo..."):
        champion, matches, qualified_by_group, ko_results = simulate_tournament(
            mapped_groups,
            features_ready,
            model
        )


    # TABLA DE FASE DE GRUPOS
    
    st.subheader("📊 Fase de grupos — Tabla de posiciones")

    group_cols = st.columns(3)
    for idx, (group_name, standings) in enumerate(qualified_by_group.items()):
        with group_cols[idx % 3]:
            st.markdown(f"#### Grupo {group_name}")
            rows = []
            for pos, (team, pts, gf, ga) in enumerate(standings):
                dg = gf - ga
                emoji = "✅" if pos < 2 else ("🟡" if pos == 2 else "❌")
                rows.append({
                    "": emoji,
                    "Equipo": display_name(team),
                    "PTS": pts,
                    "GF": gf,
                    "GC": ga,
                    "DG": dg,
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.caption("✅ Clasificado directo | 🟡 Posible mejor tercero | ❌ Eliminado")

    st.markdown("---")


    # RESULTADOS FASE DE GRUPOS
    
    with st.expander("📋 Ver resultados detallados — Fase de grupos", expanded=False):
        match_cols = st.columns(3)
        for idx, (group, games) in enumerate(matches.items()):
            with match_cols[idx % 3]:
                st.markdown(f"**Grupo {group}**")
                for g in games:
                    st.write(g)

    st.markdown("---")

    
    # BRACKET FASE KO 

    st.subheader("🏆 Fase eliminatoria — Bracket oficial FIFA 2026")

    ROUND_ORDER = [
        "Ronda de 32",
        "Octavos de Final",
        "Cuartos de Final",
        "Semifinales",
        "Final",
    ]
    ROUND_EMOJI = {
        "Ronda de 32":      "⚽",
        "Octavos de Final": "🔥",
        "Cuartos de Final": "⚡",
        "Semifinales":      "🌟",
        "Final":            "🏆",
    }

    def match_card(m, compact=False):
        """Renderiza una tarjeta de partido HTML."""
        ta  = display_name(m["team_a"])
        tb  = display_name(m["team_b"])
        la  = m.get("label_a", "")
        lb  = m.get("label_b", "")
        sa  = m["score_a"]
        sb  = m["score_b"]
        w   = m["winner"]
        pen = "✏️ pen." if m["penalties"] else ""

        win_a = w == m["team_a"]

        badge_a = f'<span style="font-size:0.7em;color:#aaa;margin-right:4px">{la}</span>' if la else ""
        badge_b = f'<span style="font-size:0.7em;color:#aaa;margin-right:4px">{lb}</span>' if lb else ""

        style_a = "font-weight:700;color:#4ade80;" if win_a  else "color:#ccc;"
        style_b = "font-weight:700;color:#4ade80;" if not win_a else "color:#ccc;"

        pen_html = f'<div style="font-size:0.7em;color:#fbbf24;margin-top:2px">{pen}</div>' if pen else ""

        return f"""
<div style="border:1px solid #334155;border-radius:8px;padding:8px 10px;
            margin-bottom:6px;background:#0f172a;min-width:180px">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:3px">
    <span style="{style_a}">{badge_a}{ta}</span>
    <span style="{style_a};font-size:1.1em;margin-left:6px">{sa}</span>
  </div>
  <div style="display:flex;justify-content:space-between;align-items:center">
    <span style="{style_b}">{badge_b}{tb}</span>
    <span style="{style_b};font-size:1.1em;margin-left:6px">{sb}</span>
  </div>
  {pen_html}
</div>"""

    # Mostrar cada ronda en un expander (Ronda de 32 colapsada por defecto)
    for rname in ROUND_ORDER:
        if rname not in ko_results:
            continue
        matches_r = ko_results[rname]
        emoji     = ROUND_EMOJI.get(rname, "")
        n_matches = len(matches_r)

        is_final = (rname == "Final")
        expanded = rname != "Ronda de 32"

        with st.expander(f"{emoji} {rname}  ({n_matches} partido{'s' if n_matches > 1 else ''})", expanded=expanded):

            if is_final:
                # Final centrada y grande
                m = matches_r[0]
                st.markdown(match_card(m), unsafe_allow_html=True)
                st.success(f"🏆 **Campeón: {display_name(m['winner'])}**")

            else:
                # Columnas adaptadas al número de partidos
                n_cols = min(n_matches, 4)
                cols   = st.columns(n_cols)
                for i, m in enumerate(matches_r):
                    with cols[i % n_cols]:
                        st.markdown(match_card(m), unsafe_allow_html=True)

    st.markdown("---")
    st.success(f"🏆 **Campeón del Mundial 2026: {display_name(champion)}**")