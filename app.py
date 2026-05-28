import os
import io
import json
import hashlib
from pathlib import Path
from datetime import date, datetime, timedelta

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from storage import (
    init_db,
    ensure_user,
    save_profile,
    load_profile,
    save_message,
    load_messages,
    save_course_material,
    load_course_context,
    load_course_materials,
    search_course_materials_keyword,
    delete_course_material,
    clear_course_materials,
    save_checkin,
    load_checkins,
    delete_user_data,
    get_user_summary,
    database_health_check,
)

from agents import run_agentic_workflow, format_workflow_debug


# ============================================================
# Ympäristöasetukset
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH, override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

try:
    if not OPENAI_API_KEY:
        OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)
except Exception:
    pass


# ============================================================
# Streamlit-asetukset
# ============================================================

st.set_page_config(
    page_title="Agenttinen opinnäytetyövalmentaja",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# Tietokannan alustus
# ============================================================

try:
    init_db()
except Exception as e:
    st.error("Tietokannan alustaminen epäonnistui.")
    st.exception(e)
    st.stop()


# ============================================================
# Apufunktiot
# ============================================================

def trunc(text: str, n: int = 12000) -> str:
    if not text:
        return ""
    return text[:n]


def safe_json(data) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def file_hash_from_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_uploaded_file(uploaded_file):
    """
    Palauttaa:
        filename: str
        file_hash: str
        text: str
    """

    filename = uploaded_file.name
    data = uploaded_file.getvalue()
    file_hash = file_hash_from_bytes(data)
    lower_name = filename.lower()

    if lower_name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        pages = []

        for page in reader.pages:
            pages.append(page.extract_text() or "")

        text = "\n".join(pages)
        return filename, file_hash, text

    if lower_name.endswith(".txt") or lower_name.endswith(".md"):
        text = data.decode("utf-8", errors="ignore")
        return filename, file_hash, text

    return filename, file_hash, ""


def parse_date_or_today(value):
    if not value:
        return date.today()

    try:
        return datetime.fromisoformat(value).date()
    except Exception:
        return date.today()


def format_recent_history(user_id: str, limit: int = 10) -> str:
    messages = load_messages(user_id, limit=limit)

    parts = []

    for msg in messages:
        parts.append(
            str(msg["created_at"])
            + " | "
            + str(msg["role"])
            + " | "
            + str(msg["mode"])
            + ":\n"
            + str(msg["content"])
        )

    return "\n\n".join(parts)


def get_relevant_course_context(user_id: str, user_input: str) -> str:
    """
    Hakee ensin osumia kurssimateriaaleista avainsanoilla.
    Jos osumia ei löydy, palauttaa yleisen kurssikontekstin.
    """

    retrieved = search_course_materials_keyword(
        user_id=user_id,
        query=user_input,
        max_results=5,
        chunk_size=1200
    )

    if retrieved.strip():
        return retrieved

    return load_course_context(user_id, max_chars=30000)


def proactive_alerts(profile: dict, checkins: list) -> list:
    alerts = []

    if not profile:
        alerts.append(
            "Luo ja tallenna opinnäytetyön profiili, jotta saat henkilökohtaisempaa tukea."
        )
        return alerts

    deadline = profile.get("deadline")
    supervisor_meeting = profile.get("supervisor_meeting")
    research_question = profile.get("research_question")
    stage = profile.get("stage")

    today = date.today()

    if deadline:
        try:
            deadline_date = datetime.fromisoformat(deadline).date()
            days_left = (deadline_date - today).days

            if days_left < 0:
                alerts.append(
                    "Tavoiteltu palautuspäivä näyttää olevan mennyt. "
                    "Päivitä aikataulu ja sovi seuraavista askeleista ohjaajan kanssa."
                )
            elif days_left <= 14:
                alerts.append(
                    "Palautuspäivään on "
                    + str(days_left)
                    + " päivää. Keskity viimeistelyyn, lähteisiin, muotoiluun ja lopputarkistuksiin."
                )
            elif days_left <= 30:
                alerts.append(
                    "Palautuspäivään on "
                    + str(days_left)
                    + " päivää. Tee tarkka viikkosuunnitelma työn loppuun saattamiseksi."
                )

        except Exception:
            pass

    if supervisor_meeting:
        try:
            meeting_date = datetime.fromisoformat(supervisor_meeting).date()
            days_until_meeting = (meeting_date - today).days

            if 0 <= days_until_meeting <= 7:
                alerts.append(
                    "Ohjaustapaaminen on pian. Valmistele lyhyt tilannekuva ja 2–3 täsmällistä kysymystä."
                )

        except Exception:
            pass

    if not research_question or len(research_question.strip()) < 30:
        alerts.append(
            "Tutkimuskysymys vaikuttaa vielä keskeneräiseltä. Sen tarkentaminen kannattaa priorisoida."
        )

    if stage in ["Aiheen valinta", "Tutkimuskysymyksen määrittely"] and checkins:
        recent_blockers = []

        for item in checkins[:3]:
            blocked = item.get("blocked")
            if blocked and len(blocked.strip()) > 10:
                recent_blockers.append(blocked)

        if len(recent_blockers) >= 2:
            alerts.append(
                "Viikkokatsaukset viittaavat toistuviin esteisiin. "
                "Keskustele rajauksesta, toteutettavuudesta ja seuraavista päätöksistä ohjaajan kanssa."
            )

    return alerts


def load_user_state(user_id: str):
    ensure_user(user_id)
    st.session_state.profile = load_profile(user_id)
    st.session_state.course_context = load_course_context(user_id)
    st.session_state.loaded_user_id = user_id


def agent_label(agent_name: str) -> str:
    labels = {
        "integrity": "akateeminen rehellisyys",
        "planner": "suunnittelu",
        "writing_coach": "kirjoituspalaute",
        "criteria_alignment": "kriteerivastaavuus",
        "research_design": "tutkimusasetelma",
        "reflection": "reflektio ja itsesäätely",
        "weekly_plan": "viikkosuunnitelma",
        "supervision_summary": "ohjausmuistio"
    }

    return labels.get(agent_name, agent_name)


def run_workflow(task_type: str, user_input: str, user_id: str, mode: str):
    recent_history = format_recent_history(user_id, limit=8)
    checkins = load_checkins(user_id, limit=5)
    course_context = get_relevant_course_context(user_id, user_input)

    result = run_agentic_workflow(
        task_type=task_type,
        user_input=user_input,
        profile=st.session_state.get("profile", {}),
        course_context=course_context,
        recent_history=recent_history,
        checkins=checkins
    )

    selected_agents = [
        agent_label(name)
        for name in result.get("selected_agents", [])
    ]

    if selected_agents:
        st.info("Valitut agentit: " + ", ".join(selected_agents))

    save_message(user_id, "user", user_input, mode=mode)
    save_message(user_id, "assistant", result["final_response"], mode=mode)

    st.markdown(result["final_response"])

    with st.expander("Agenttisen työnkulun tekniset tiedot"):
        st.markdown(format_workflow_debug(result))

    return result


def add_demo_profile(user_id: str):
    demo_profile = {
        "discipline": "Kasvatustiede",
        "topic": "Tekoälyn käyttö opinnäytetyön kirjoittamisen tukena",
        "title": "Opiskelijoiden kokemuksia tekoälyavusteisesta opinnäytetyöprosessista",
        "research_question": (
            "Miten opiskelijat kuvaavat tekoälypohjaisen kirjoittamisen tuen vaikutusta "
            "opinnäytetyöprosessinsa suunnitteluun ja etenemiseen?"
        ),
        "thesis_type": "Empiirinen laadullinen tutkimus",
        "method": "Laadullinen haastattelututkimus, temaattinen analyysi",
        "stage": "Tutkimuskysymyksen määrittely",
        "deadline": str(date.today() + timedelta(days=30)),
        "supervisor_meeting": str(date.today() + timedelta(days=6)),
        "current_challenge": (
            "Tutkimuskysymys on vielä melko laaja ja aineistonkeruun rajaus on epäselvä."
        )
    }

    st.session_state.profile = demo_profile
    save_profile(user_id, demo_profile)
    save_message(
        user_id=user_id,
        role="system",
        content="Demoprofiili lisätty:\n" + safe_json(demo_profile),
        mode="demo_profile"
    )


def add_demo_course_material(user_id: str):
    demo_material = """
OPINNÄYTETYÖN TAVOITTEET JA ARVIOINTIKRITEERIT

Hyvä opinnäytetyö:
- esittää selkeän ja rajatun tutkimuskysymyksen,
- perustelee aiheen merkityksen aiemman tutkimuksen avulla,
- valitsee tutkimuskysymykseen sopivan aineiston ja menetelmän,
- kuvaa aineistonkeruun ja analyysin läpinäkyvästi,
- noudattaa hyvää tieteellistä käytäntöä,
- arvioi tutkimuksen luotettavuutta ja eettisiä kysymyksiä,
- rakentaa johdonmukaisen argumentin,
- käyttää lähteitä asianmukaisesti.

Tutkimussuunnitelmassa tulisi kuvata:
- tutkimuksen aihe ja tausta,
- alustava tutkimuskysymys,
- aineisto tai tutkimusmateriaali,
- menetelmä ja analyysitapa,
- alustava aikataulu,
- mahdolliset eettiset kysymykset,
- seuraavat päätökset, joista tarvitaan ohjaajan palautetta.

Ohjaustapaamiseen valmistautuminen:
- tiivistä eteneminen lyhyesti,
- nimeä 1–3 konkreettista ongelmaa,
- ehdota vaihtoehtoja, joista tarvitset palautetta,
- kerro, mikä päätös pitäisi tehdä seuraavaksi.

Akateemisen kirjoittamisen näkökulmasta tekstin tulisi:
- edetä loogisesti,
- erottaa tutkimuksen tausta, tavoite, aineisto, menetelmä ja analyysi,
- perustella väitteet lähteillä,
- välttää liian yleisiä väitteitä,
- käyttää täsmällisiä käsitteitä.
"""

    save_course_material(
        user_id=user_id,
        filename="demokurssimateriaali.txt",
        content=demo_material,
        file_hash="demo-material-v1"
    )

    st.session_state.course_context = load_course_context(user_id)


# ============================================================
# Otsikko ja demoa tukevat kuvausosiot
# ============================================================

st.title("Agenttinen tekoäly opinnäytetyön tueksi")

st.caption(
    "Pilvipohjainen Streamlit-prototyyppi, jossa orkestroija-agentti ja erikoistuneet agentit "
    "tukevat suunnittelua, kirjoittamista, tutkimusasetelmaa, reflektiota ja ohjaukseen valmistautumista."
)

with st.expander("Mikä tämä prototyyppi on?", expanded=True):
    st.markdown("""
Tämä prototyyppi on **agenttinen opinnäytetyövalmentaja** korkeakouluopiskelijoille.

Sen tarkoitus on tukea opiskelijaa:
- opinnäytetyöprosessin suunnittelussa,
- tutkimuskysymyksen ja tutkimusasetelman jäsentämisessä,
- tekstiluonnosten formatiivisessa palautteessa,
- viikoittaisessa etenemisen seurannassa,
- ohjaustapaamisiin valmistautumisessa.

**Agenttisuus tarkoittaa tässä**, että yksi vastaus muodostuu usean erikoistuneen agentin yhteistyönä:
- orkestroija valitsee tarvittavat agentit,
- kirjoitusagentti arvioi tekstiä,
- tutkimusasetelma-agentti arvioi menetelmän ja rajauksen suhdetta,
- kriteeriagentti vertaa tuotosta annettuihin kurssimateriaaleihin,
- reflektioagentti tukee opiskelijan itsesäätelyä,
- lopullinen vastaus koostetaan opiskelijalle ymmärrettävään muotoon.

**Rajaus:** työkalu ei kirjoita opinnäytetyötä opiskelijan puolesta eikä korvaa ohjaajaa.
""")

with st.expander("Tietosuoja ja tarkoituksenmukainen käyttö", expanded=False):
    st.markdown(
        "Tämä on opinnäytetyön ohjaukselliseen tukeen tarkoitettu prototyyppi.\n\n"
        "**Tallennettavat tiedot**\n"
        "- pseudonyymi käyttäjätunnus\n"
        "- opinnäytetyön profiili\n"
        "- ladatut kurssimateriaalit\n"
        "- keskusteluhistoria\n"
        "- viikoittaiset tilannekatsaukset\n\n"
        "**Älä lataa palveluun**\n"
        "- oikeita nimiä tai opiskelijanumeroita\n"
        "- luottamuksellista tutkimusaineistoa\n"
        "- tunnistettavia henkilötietoja\n"
        "- arkaluonteista tai julkaisematonta aineistoa\n"
        "- ohjaajan kommentteja, joita ei ole tarkoitettu jaettavaksi\n\n"
        "**Akateeminen rehellisyys**\n"
        "Työkalu tukee suunnittelua, palautetta ja reflektiota. "
        "Sitä ei tule käyttää opinnäytetyön kirjoittamiseen opiskelijan puolesta.\n\n"
        "Todellisessa yliopistokäytössä tulee tarkistaa tietosuoja, tutkimuseettiset vaatimukset, "
        "tekoälyn käyttöä koskevat ohjeet, hankintakäytännöt ja organisaation sisäiset linjaukset."
    )

with st.expander("Tunnetut rajoitukset", expanded=False):
    st.markdown("""
- Prototyyppi ei korvaa opinnäytetyön ohjaajaa.
- Kurssimateriaalien haku on vielä avainsanapohjainen, ei semanttinen.
- Malli voi tehdä virheitä tai antaa liian yleisiä suosituksia.
- Prototyyppi ei sovellu arkaluonteisen tai tunnistettavan tutkimusaineiston käsittelyyn.
- Käyttöönotto oikeilla opiskelijoilla edellyttää tietosuoja- ja tekoälyohjeiden tarkistamista.
- Opiskelijan tulee itse arvioida ja muokata kaikki tuotokset ennen käyttöä.
""")

with st.expander("Miten tätä voisi arvioida pilotissa?", expanded=False):
    st.markdown("""
Mahdollisia arviointikohteita:
- kokevatko opiskelijat saavansa apua opinnäytetyön suunnitteluun,
- tarkentuvatko tutkimuskysymykset ja seuraavat askeleet,
- valmistautuvatko opiskelijat paremmin ohjaustapaamisiin,
- väheneekö ohjaajalle tulevien yleisten prosessikysymysten määrä,
- säilyykö akateeminen rehellisyys,
- perustuuko palaute annettuihin kurssimateriaaleihin,
- millaisia virheitä tai liian vahvoja suosituksia järjestelmä tuottaa.
""")


# ============================================================
# Sivupalkki: käyttäjä
# ============================================================

st.sidebar.header("Käyttäjä")

user_id = st.sidebar.text_input(
    "Pseudonyymi käyttäjätunnus",
    value=st.session_state.get("loaded_user_id", "demo-opiskelija"),
    help="Käytä pseudonyymiä tunnusta. Vältä oikeita nimiä, opiskelijanumeroita ja arkaluonteisia tunnisteita."
).strip()

if not user_id:
    st.sidebar.error("Anna käyttäjätunnus.")
    st.stop()

if st.session_state.get("loaded_user_id") != user_id:
    load_user_state(user_id)

if st.sidebar.button("Lataa käyttäjän tiedot uudelleen"):
    load_user_state(user_id)
    st.sidebar.success("Tiedot ladattu.")


# ============================================================
# Sivupalkki: järjestelmän tila
# ============================================================

with st.sidebar.expander("Järjestelmän tila", expanded=False):
    health = database_health_check()

    if health["ok"]:
        st.success(health["message"])
    else:
        st.error(health["message"])

    st.write("OpenAI API -avain löytyi:", bool(OPENAI_API_KEY))

summary = get_user_summary(user_id)

st.sidebar.markdown("### Tallennetut tiedot")
st.sidebar.write("Viestit:", summary["message_count"])
st.sidebar.write("Kurssimateriaalit:", summary["material_count"])
st.sidebar.write("Viikkokatsaukset:", summary["checkin_count"])


# ============================================================
# Sivupalkki: demopainikkeet
# ============================================================

st.sidebar.markdown("---")
st.sidebar.header("Demo")

if st.sidebar.button("Täytä demoprofiili"):
    add_demo_profile(user_id)
    st.sidebar.success("Demoprofiili lisätty.")
    st.rerun()

if st.sidebar.button("Lisää demokurssimateriaali"):
    add_demo_course_material(user_id)
    st.sidebar.success("Demokurssimateriaali lisätty.")
    st.rerun()


# ============================================================
# Sivupalkki: kurssimateriaalit
# ============================================================

st.sidebar.markdown("---")
st.sidebar.header("Kurssimateriaalit ja tavoitteet")

uploaded_files = st.sidebar.file_uploader(
    "Lataa opinnäytetyöohjeet, arviointikriteerit, kirjoitusohje tai kurssin tavoitteet",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True
)

if st.sidebar.button("Tallenna ladatut materiaalit"):
    if not uploaded_files:
        st.sidebar.warning("Ei valittuja tiedostoja.")
    else:
        inserted_count = 0
        duplicate_count = 0
        unreadable_count = 0

        for uploaded_file in uploaded_files:
            filename, file_hash, text = read_uploaded_file(uploaded_file)

            if not text.strip():
                unreadable_count += 1
                st.sidebar.warning("Tiedostosta ei saatu luettavaa tekstiä: " + filename)
                continue

            inserted = save_course_material(
                user_id=user_id,
                filename=filename,
                content=text,
                file_hash=file_hash
            )

            if inserted:
                inserted_count += 1
            else:
                duplicate_count += 1

        st.session_state.course_context = load_course_context(user_id)

        st.sidebar.success(
            "Tallennettu: "
            + str(inserted_count)
            + ", duplikaatteja: "
            + str(duplicate_count)
            + ", ei luettavissa: "
            + str(unreadable_count)
        )

materials = load_course_materials(user_id)

with st.sidebar.expander("Tallennetut materiaalit", expanded=False):
    if not materials:
        st.write("Ei tallennettuja materiaaleja.")
    else:
        for item in materials:
            cols = st.columns([3, 1])

            with cols[0]:
                st.write(item["filename"])
                st.caption(item["created_at"])

            with cols[1]:
                if st.button("Poista", key="delete_material_" + str(item["id"])):
                    delete_course_material(user_id, item["id"])
                    st.session_state.course_context = load_course_context(user_id)
                    st.rerun()

if st.sidebar.button("Tyhjennä kaikki materiaalit"):
    clear_course_materials(user_id)
    st.session_state.course_context = ""
    st.sidebar.info("Materiaalit poistettu.")
    st.rerun()

with st.sidebar.expander("Materiaalikontekstin esikatselu", expanded=False):
    st.write(
        trunc(st.session_state.get("course_context", ""), 2000)
        or "Ei ladattua kontekstia."
    )


# ============================================================
# Sivupalkki: tietojen poisto
# ============================================================

st.sidebar.markdown("---")
st.sidebar.header("Tietojen poisto")

st.sidebar.warning(
    "Prototyyppi. Älä lataa luottamuksellista tutkimusaineistoa, henkilötietoja, "
    "opiskelijanumeroita tai tunnistettavia osallistujatietoja."
)

confirm_delete = st.sidebar.checkbox("Ymmärrän, että poisto on pysyvä")

if st.sidebar.button("Poista kaikki tietoni"):
    if confirm_delete:
        delete_user_data(user_id)
        st.session_state.clear()
        st.sidebar.success("Käyttäjän tiedot poistettu.")
        st.rerun()
    else:
        st.sidebar.error("Vahvista poisto ensin.")


# ============================================================
# Ennakoivat huomiot
# ============================================================

profile = st.session_state.get("profile", {})
checkins = load_checkins(user_id, limit=10)
alerts = proactive_alerts(profile, checkins)

if alerts:
    st.warning("Ennakoivat huomiot")
    for alert in alerts:
        st.write("- " + alert)


# ============================================================
# Välilehdet
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "1. Opinnäytetyön profiili",
    "2. Ennakoiva valmentaja",
    "3. Tekstipalaute",
    "4. Tutkimusasetelma",
    "5. Viikkokatsaus",
    "6. Viikkosuunnitelma",
    "7. Ohjaukseen valmistautuminen",
    "8. Historia"
])


# ============================================================
# Tab 1: profiili
# ============================================================

with tab1:
    st.header("Opinnäytetyön profiili")

    if st.button("Täytä demoprofiili tähän näkymään", key="demo_profile_tab"):
        add_demo_profile(user_id)
        st.success("Demoprofiili täytetty ja tallennettu.")
        st.rerun()

    current = st.session_state.get("profile", {})

    discipline = st.text_input(
        "Oppiaine tai koulutusohjelma",
        value=current.get("discipline", "")
    )

    topic = st.text_input(
        "Alustava aihe",
        value=current.get("topic", "")
    )

    title = st.text_input(
        "Alustava otsikko",
        value=current.get("title", "")
    )

    research_question = st.text_area(
        "Nykyinen tutkimuskysymys",
        value=current.get("research_question", ""),
        height=120
    )

    thesis_type_options = [
        "En ole varma",
        "Kirjallisuuskatsaus",
        "Empiirinen laadullinen tutkimus",
        "Empiirinen määrällinen tutkimus",
        "Monimenetelmällinen tutkimus",
        "Suunnittelu- tai kehittämistyö",
        "Muu"
    ]

    current_thesis_type = current.get("thesis_type", "En ole varma")

    if current_thesis_type in thesis_type_options:
        thesis_type_index = thesis_type_options.index(current_thesis_type)
    else:
        thesis_type_index = 0

    thesis_type = st.selectbox(
        "Opinnäytetyön tyyppi",
        thesis_type_options,
        index=thesis_type_index
    )

    method = st.text_area(
        "Suunniteltu menetelmä tai lähestymistapa",
        value=current.get("method", ""),
        height=120
    )

    stage_options = [
        "Aiheen valinta",
        "Tutkimuskysymyksen määrittely",
        "Menetelmän suunnittelu",
        "Aineiston keruu",
        "Aineiston analyysi",
        "Ensimmäisen luonnoksen kirjoittaminen",
        "Luonnoksen muokkaaminen",
        "Loppupalautukseen valmistautuminen"
    ]

    current_stage = current.get("stage", "Aiheen valinta")

    if current_stage in stage_options:
        stage_index = stage_options.index(current_stage)
    else:
        stage_index = 0

    stage = st.selectbox(
        "Nykyinen vaihe",
        stage_options,
        index=stage_index
    )

    deadline = st.date_input(
        "Tavoiteltu palautuspäivä",
        value=parse_date_or_today(current.get("deadline"))
    )

    supervisor_meeting = st.date_input(
        "Seuraava ohjaustapaaminen",
        value=parse_date_or_today(current.get("supervisor_meeting"))
    )

    current_challenge = st.text_area(
        "Suurin tämänhetkinen haaste",
        value=current.get("current_challenge", ""),
        height=120
    )

    if st.button("Tallenna profiili"):
        updated_profile = {
            "discipline": discipline,
            "topic": topic,
            "title": title,
            "research_question": research_question,
            "thesis_type": thesis_type,
            "method": method,
            "stage": stage,
            "deadline": str(deadline),
            "supervisor_meeting": str(supervisor_meeting),
            "current_challenge": current_challenge,
        }

        st.session_state.profile = updated_profile
        save_profile(user_id, updated_profile)

        save_message(
            user_id=user_id,
            role="system",
            content="Opinnäytetyön profiili päivitetty:\n" + safe_json(updated_profile),
            mode="profile_update"
        )

        st.success("Profiili tallennettu.")

    st.subheader("Tallennettu profiili")
    st.json(st.session_state.get("profile", {}))


# ============================================================
# Tab 2: ennakoiva valmentaja
# ============================================================

with tab2:
    st.header("Ennakoiva opinnäytetyövalmentaja")

    st.write(
        "Kuvaa tilannettasi, niin orkestroija-agentti valitsee sopivat erikoisagentit "
        "ja muodostaa sinulle seuraavat askeleet."
    )

    if "coach_input_demo" not in st.session_state:
        st.session_state.coach_input_demo = ""

    if st.button("Käytä demotilannetta", key="demo_coach_input"):
        st.session_state.coach_input_demo = (
            "Minulla on aihe ja alustava tutkimuskysymys, mutta en ole varma, "
            "onko tutkimuskysymys liian laaja. Ohjaustapaaminen on ensi viikolla, "
            "ja haluaisin tietää, mitä minun kannattaa valmistella ennen tapaamista."
        )

    coach_input = st.text_area(
        "Kuvaa tämänhetkinen tilanteesi",
        value=st.session_state.coach_input_demo,
        height=180,
        key="coach_input_area"
    )

    if st.button("Pyydä ennakoivaa valmennusta"):
        if not st.session_state.get("profile"):
            st.warning("Tallenna ensin opinnäytetyön profiili.")
        elif not coach_input.strip():
            st.warning("Kuvaa ensin tilanteesi.")
        else:
            with st.spinner("Orkestroija valitsee agentit ja koostaa palautteen. Tämä voi kestää hetken..."):
                run_workflow("proactive_coach", coach_input, user_id, "proactive_coach")


# ============================================================
# Tab 3: tekstipalaute
# ============================================================

with tab3:
    st.header("Tekstipalaute")

    st.write(
        "Liitä kappale, rakenne, johdanto, menetelmäosio tai suunnitelma. "
        "Agentit antavat formatiivista palautetta kirjoittamatta tekstiä puolestasi."
    )

    if "draft_text_demo" not in st.session_state:
        st.session_state.draft_text_demo = ""

    if st.button("Käytä demotekstiä", key="demo_draft_input"):
        st.session_state.draft_text_demo = (
            "Tässä tutkielmassa tarkastelen tekoälyn käyttöä opiskelussa. "
            "Tekoäly on nykyään tärkeä aihe, ja monet opiskelijat käyttävät sitä. "
            "Tutkimukseni selvittää, miten tekoäly vaikuttaa opiskelijoihin. "
            "Aineisto kerätään haastatteluilla ja analysoidaan jotenkin laadullisesti."
        )

    draft_text = st.text_area(
        "Liitä teksti tähän",
        value=st.session_state.draft_text_demo,
        height=320,
        key="draft_text_area"
    )

    if st.button("Anna palautetta tekstistä"):
        if not draft_text.strip():
            st.warning("Liitä ensin tekstiä.")
        else:
            with st.spinner("Kirjoitus-, kriteeri- ja rehellisyysagentit työskentelevät..."):
                run_workflow("draft_feedback", draft_text, user_id, "draft_feedback")


# ============================================================
# Tab 4: tutkimusasetelma
# ============================================================

with tab4:
    st.header("Tutkimusasetelman tuki")

    if "design_input_demo" not in st.session_state:
        st.session_state.design_input_demo = ""

    if st.button("Käytä demotutkimusasetelmaa", key="demo_design_input"):
        st.session_state.design_input_demo = (
            "Tutkimuskysymykseni on: Miten opiskelijat käyttävät tekoälyä opinnäytetyön tekemisessä? "
            "Ajattelen kerätä aineiston 5–6 opiskelijan haastatteluilla. "
            "Menetelmänä voisi olla laadullinen haastattelututkimus ja analyysina temaattinen analyysi. "
            "En ole vielä varma, pitäisikö rajata aihe kirjoittamisen suunnitteluun, tekstipalautteeseen "
            "vai koko opinnäytetyöprosessiin."
        )

    design_input = st.text_area(
        "Kuvaa tutkimuskysymys, aineisto, menetelmä ja suunniteltu analyysi",
        value=st.session_state.design_input_demo,
        height=280,
        key="design_input_area"
    )

    if st.button("Analysoi tutkimusasetelma"):
        if not design_input.strip():
            st.warning("Kuvaa ensin tutkimusasetelmasi.")
        else:
            with st.spinner("Tutkimusasetelma-agentit työskentelevät..."):
                run_workflow("research_design", design_input, user_id, "research_design")


# ============================================================
# Tab 5: viikkokatsaus
# ============================================================

with tab5:
    st.header("Viikoittainen tilannekatsaus")

    completed = st.text_area("Mitä sait tällä viikolla valmiiksi?", height=100)
    blocked = st.text_area("Mikä estää etenemistä?", height=100)
    next_action = st.text_area("Mitä aiot tehdä seuraavaksi?", height=100)
    support_needed = st.text_area("Millaista tukea tarvitset?", height=100)

    if st.button("Täytä demoviikkokatsaus", key="demo_weekly_checkin"):
        completed_demo = "Tarkensin aihetta ja luin kolme aiheeseen liittyvää artikkelia."
        blocked_demo = "En ole varma, miten rajaan tutkimuskysymyksen riittävän kapeaksi."
        next_action_demo = "Haluan laatia kaksi vaihtoehtoista tutkimuskysymystä ohjaajalle."
        support_demo = "Tarvitsen apua rajauksen ja seuraavien konkreettisten tehtävien määrittelyssä."

        checkin_text = (
            "Valmistui tällä viikolla:\n"
            + completed_demo
            + "\n\nEsteet:\n"
            + blocked_demo
            + "\n\nSeuraava suunniteltu teko:\n"
            + next_action_demo
            + "\n\nTarvittava tuki:\n"
            + support_demo
        )

        with st.spinner("Viikkokatsausagentit työskentelevät..."):
            result = run_workflow(
                "weekly_checkin",
                checkin_text,
                user_id,
                "weekly_checkin"
            )

        save_checkin(
            user_id=user_id,
            completed=completed_demo,
            blocked=blocked_demo,
            next_action=next_action_demo,
            support_needed=support_demo,
            coach_response=result["final_response"]
        )

    if st.button("Luo viikkovalmentajan vastaus"):
        if not any([
            completed.strip(),
            blocked.strip(),
            next_action.strip(),
            support_needed.strip()
        ]):
            st.warning("Kirjoita ensin vähintään yksi kohta.")
        else:
            checkin_text = (
                "Valmistui tällä viikolla:\n"
                + completed
                + "\n\nEsteet:\n"
                + blocked
                + "\n\nSeuraava suunniteltu teko:\n"
                + next_action
                + "\n\nTarvittava tuki:\n"
                + support_needed
            )

            with st.spinner("Viikkokatsausagentit työskentelevät..."):
                result = run_workflow(
                    "weekly_checkin",
                    checkin_text,
                    user_id,
                    "weekly_checkin"
                )

            save_checkin(
                user_id=user_id,
                completed=completed,
                blocked=blocked,
                next_action=next_action,
                support_needed=support_needed,
                coach_response=result["final_response"]
            )

    st.subheader("Aiemmat viikkokatsaukset")

    previous_checkins = load_checkins(user_id, limit=10)

    if not previous_checkins:
        st.write("Ei aiempia viikkokatsauksia.")
    else:
        for item in previous_checkins:
            with st.expander("Viikkokatsaus " + str(item["created_at"])):
                st.markdown("**Valmistui**")
                st.write(item["completed"] or "-")

                st.markdown("**Esteet**")
                st.write(item["blocked"] or "-")

                st.markdown("**Seuraava teko**")
                st.write(item["next_action"] or "-")

                st.markdown("**Tarvittava tuki**")
                st.write(item["support_needed"] or "-")

                st.markdown("**Valmentajan vastaus**")
                st.markdown(item["coach_response"] or "-")


# ============================================================
# Tab 6: viikkosuunnitelma
# ============================================================

with tab6:
    st.header("Seitsemän päivän viikkosuunnitelma")

    st.write(
        "Luo realistinen seitsemän päivän suunnitelma profiilisi, aikataulusi, "
        "viikkokatsaustesi ja kurssimateriaalien perusteella."
    )

    if "weekly_input_demo" not in st.session_state:
        st.session_state.weekly_input_demo = ""

    if st.button("Käytä demopyyntöä viikkosuunnitelmaan", key="demo_weekly_plan"):
        st.session_state.weekly_input_demo = (
            "Minulla on tällä viikolla noin 8 tuntia aikaa. "
            "Haluan valmistella ohjaustapaamista varten tutkimuskysymyksen rajauksen "
            "ja alustavan menetelmäkuvauksen."
        )

    weekly_input = st.text_area(
        "Valinnainen tarkennus suunnitelmalle",
        value=st.session_state.weekly_input_demo,
        height=160,
        key="weekly_input_area"
    )

    if st.button("Luo 7 päivän suunnitelma"):
        if not st.session_state.get("profile"):
            st.warning("Tallenna ensin opinnäytetyön profiili.")
        else:
            if weekly_input.strip():
                user_input = weekly_input.strip()
            else:
                user_input = (
                    "Laadi realistinen 7 päivän opinnäytetyösuunnitelma profiilini, "
                    "aikatauluni ja aiempien viikkokatsausten perusteella."
                )

            with st.spinner("Viikkosuunnitelma-agentit työskentelevät..."):
                run_workflow("weekly_plan", user_input, user_id, "weekly_plan")


# ============================================================
# Tab 7: ohjaukseen valmistautuminen
# ============================================================

with tab7:
    st.header("Ohjaukseen valmistautuminen")

    st.write(
        "Luo tiivis muistio ohjaustapaamista varten. Tarkista ja muokkaa muistio ennen jakamista."
    )

    if "supervision_input_demo" not in st.session_state:
        st.session_state.supervision_input_demo = ""

    if st.button("Käytä demo-ohjaustilannetta", key="demo_supervision_input"):
        st.session_state.supervision_input_demo = (
            "Haluan keskustella ohjaajan kanssa siitä, onko tutkimuskysymykseni liian laaja, "
            "riittääkö 5–6 haastattelua aineistoksi ja kannattaako analyysitavaksi valita temaattinen analyysi. "
            "Tarvitsen myös päätöksen siitä, mitä teen seuraavaksi ennen tutkimussuunnitelman palautusta."
        )

    supervision_input = st.text_area(
        "Kuvaa, mitä haluat käsitellä ohjaajan kanssa",
        value=st.session_state.supervision_input_demo,
        height=180,
        key="supervision_input_area"
    )

    if st.button("Luo ohjaustapaamisen muistio"):
        if not st.session_state.get("profile"):
            st.warning("Tallenna ensin opinnäytetyön profiili.")
        else:
            if supervision_input.strip():
                user_input = supervision_input.strip()
            else:
                user_input = (
                    "Valmistele ohjaustapaamisen muistio profiilini, esteideni "
                    "ja seuraavien päätösten perusteella."
                )

            with st.spinner("Ohjaukseen valmistautumisen agentit työskentelevät..."):
                run_workflow(
                    "supervision_summary",
                    user_input,
                    user_id,
                    "supervision_summary"
                )


# ============================================================
# Tab 8: historia
# ============================================================

with tab8:
    st.header("Tallennettu keskusteluhistoria")

    limit = st.slider("Näytettävien viestien määrä", 5, 100, 30)

    messages = load_messages(user_id, limit=limit)

    if not messages:
        st.write("Ei tallennettuja viestejä.")
    else:
        for msg in messages:
            role = msg["role"]

            if role == "user":
                chat_role = "user"
            else:
                chat_role = "assistant"

            with st.chat_message(chat_role):
                st.markdown(msg["content"])
                st.caption(
                    str(msg["mode"] or "yleinen")
                    + " | "
                    + str(msg["created_at"])
                )

    st.markdown("---")

    if st.button("Päivitä historia"):
        st.rerun()
