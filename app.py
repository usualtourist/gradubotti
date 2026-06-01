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
        "orchestrator": "Työnkulun ohjaaja",
        "integrity": "Rehellisyysvahti",
        "planner": "Etenemisluotsi",
        "writing_coach": "Tekstiluotsi",
        "criteria_alignment": "Kriteeriluotsi",
        "research_design": "Tutkimusluotsi",
        "reflection": "Reflektiokumppani",
        "weekly_plan": "Viikkovalmentaja",
        "supervision_summary": "Ohjaustapaamisen valmistelija",
        "finalizer": "Vastauskoostaja"
    }

    return labels.get(agent_name, agent_name)


def agent_description(agent_name: str) -> str:
    descriptions = {
        "orchestrator": (
            "Analysoi opiskelijan pyynnön ja valitsee, mitä erikoisagentteja tarvitaan."
        ),
        "integrity": (
            "Tarkistaa, että tuki pysyy akateemisen rehellisyyden rajoissa eikä työkalu "
            "tee opinnäytetyötä opiskelijan puolesta."
        ),
        "planner": (
            "Jäsentää opiskelijan tilanteen, seuraavat konkreettiset askeleet, aikataulun "
            "ja ohjaajalle vietävät kysymykset."
        ),
        "writing_coach": (
            "Antaa formatiivista palautetta tekstin selkeydestä, rakenteesta, argumentaatiosta "
            "ja akateemisesta tyylistä."
        ),
        "criteria_alignment": (
            "Vertaa opiskelijan suunnitelmaa tai tekstiä annettuihin kurssimateriaaleihin, "
            "tavoitteisiin ja arviointikriteereihin."
        ),
        "research_design": (
            "Arvioi tutkimuskysymyksen, aineiston, menetelmän, analyysin ja rajauksen "
            "yhteensopivuutta."
        ),
        "reflection": (
            "Tukee opiskelijan itsesäätelyä, etenemisen arviointia ja seuraavan pienen "
            "askeleen valintaa."
        ),
        "weekly_plan": (
            "Laatii realistisen seitsemän päivän etenemissuunnitelman."
        ),
        "supervision_summary": (
            "Koostaa opiskelijan tarkistettavan muistion ohjaustapaamista varten."
        ),
        "finalizer": (
            "Yhdistää erikoisagenttien havainnot yhdeksi opiskelijalle selkeäksi vastaukseksi."
        )
    }

    return descriptions.get(agent_name, "")


def run_workflow(task_type: str, user_input: str, user_id: str, mode: str):
    recent_history = format_recent_history(user_id, limit=8)
    checkins = load_checkins(user_id, limit=5)
    course_context = get_relevant_course_context(user_id, user_input)
    course_context_used = bool(course_context and course_context.strip())

    if course_context_used:
        st.success("Kurssimateriaalia käytetään tämän vastauksen tukena.")
    else:
        st.info(
            "Kurssimateriaalia ei löytynyt tähän pyyntöön. "
            "Vastaus perustuu profiiliin, keskusteluhistoriaan ja yleiseen ohjaukselliseen tukeen."
        )

    progress_placeholder = st.empty()
    progress_events = []
    selected_agent_names = []
    agent_output_previews = []

    def render_progress():
        with progress_placeholder.container():
            st.markdown("### Agenttinen työnkulku")

            st.markdown(
                "Tämä vastaus muodostetaan useassa vaiheessa:\n\n"
                "1. **Työnkulun ohjaaja** analysoi opiskelijan pyynnön.\n"
                "2. Työnkulun ohjaaja valitsee tarvittavat **erikoisagentit**.\n"
                "3. Erikoisagentit tarkastelevat tilannetta eri näkökulmista.\n"
                "4. **Vastauskoostaja** yhdistää tulokset opiskelijalle suunnatuksi vastaukseksi."
            )

            if selected_agent_names:
                st.markdown("#### Valitut agentit")

                for agent_name in selected_agent_names:
                    st.markdown(
                        "- **"
                        + agent_label(agent_name)
                        + "**: "
                        + agent_description(agent_name)
                    )

            st.markdown("#### Eteneminen")

            for item in progress_events:
                st.markdown(item)

            if agent_output_previews:
                with st.expander("Näytä agenttien lyhyet välitulokset", expanded=False):
                    st.markdown(
                        "Nämä ovat tiivistettyjä välituloksia. Lopullinen opiskelijalle tarkoitettu "
                        "vastaus muodostetaan niiden perusteella erikseen."
                    )

                    for preview in agent_output_previews:
                        st.markdown(preview)

    def progress_callback(update: dict):
        event = update.get("event")
        agent_name = update.get("agent")
        label = agent_label(agent_name) if agent_name else update.get("label", "Agentti")
        message = update.get("message", "")

        if event == "workflow_start":
            progress_events.append("🟦 Työnkulku käynnistyi.")

        elif event == "orchestrator_start":
            progress_events.append("⏳ **Työnkulun ohjaaja** analysoi pyynnön.")

        elif event == "route_complete":
            data = update.get("data") or {}
            agents = data.get("selected_agents", [])

            selected_agent_names.clear()
            selected_agent_names.extend(agents)

            progress_events.append("✅ **Työnkulun ohjaaja valitsi tarvittavat agentit.**")

        elif event == "agent_start":
            progress_events.append("⏳ **" + label + "** käsittelee pyyntöä.")

        elif event == "agent_complete":
            progress_events.append("✅ **" + label + "** valmis.")

            data = update.get("data") or {}
            output_preview = data.get("output_preview")

            if output_preview:
                short_preview = output_preview[:500]

                agent_output_previews.append(
                    "#### "
                    + label
                    + "\n\n"
                    + short_preview
                    + "\n\n"
                )

        elif event == "finalizer_start":
            progress_events.append("⏳ **Vastauskoostaja** muodostaa lopullisen vastauksen.")

        elif event == "finalizer_complete":
            progress_events.append("✅ **Vastauskoostaja** valmis.")

        elif event == "workflow_complete":
            progress_events.append("🟩 **Agenttinen työnkulku valmis.**")

        elif message:
            progress_events.append("ℹ️ " + message)

        render_progress()

    render_progress()

    result = run_agentic_workflow(
        task_type=task_type,
        user_input=user_input,
        profile=st.session_state.get("profile", {}),
        course_context=course_context,
        recent_history=recent_history,
        checkins=checkins,
        progress_callback=progress_callback
    )

    selected_agents = [
        agent_label(name)
        for name in result.get("selected_agents", [])
    ]

    if selected_agents:
        st.info("Valitut agentit: " + ", ".join(selected_agents))

    save_message(user_id, "user", user_input, mode=mode)
    save_message(user_id, "assistant", result["final_response"], mode=mode)

    st.markdown("---")
    st.markdown("## Lopullinen vastaus")
    
    st.markdown(result["final_response"])

    with st.expander("Lisätiedot: agenttien tekniset välitulokset", expanded=False):
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
    demo_material = (
        "OPINNÄYTETYÖN TAVOITTEET JA ARVIOINTIKRITEERIT\n\n"
        "Hyvä opinnäytetyö:\n"
        "- esittää selkeän ja rajatun tutkimuskysymyksen,\n"
        "- perustelee aiheen merkityksen aiemman tutkimuksen avulla,\n"
        "- valitsee tutkimuskysymykseen sopivan aineiston ja menetelmän,\n"
        "- kuvaa aineistonkeruun ja analyysin läpinäkyvästi,\n"
        "- noudattaa hyvää tieteellistä käytäntöä,\n"
        "- arvioi tutkimuksen luotettavuutta ja eettisiä kysymyksiä,\n"
        "- rakentaa johdonmukaisen argumentin,\n"
        "- käyttää lähteitä asianmukaisesti.\n\n"
        "Tutkimussuunnitelmassa tulisi kuvata:\n"
        "- tutkimuksen aihe ja tausta,\n"
        "- alustava tutkimuskysymys,\n"
        "- aineisto tai tutkimusmateriaali,\n"
        "- menetelmä ja analyysitapa,\n"
        "- alustava aikataulu,\n"
        "- mahdolliset eettiset kysymykset,\n"
        "- seuraavat päätökset, joista tarvitaan ohjaajan palautetta.\n\n"
        "Ohjaustapaamiseen valmistautuminen:\n"
        "- tiivistä eteneminen lyhyesti,\n"
        "- nimeä 1–3 konkreettista ongelmaa,\n"
        "- ehdota vaihtoehtoja, joista tarvitset palautetta,\n"
        "- kerro, mikä päätös pitäisi tehdä seuraavaksi.\n\n"
        "Akateemisen kirjoittamisen näkökulmasta tekstin tulisi:\n"
        "- edetä loogisesti,\n"
        "- erottaa tutkimuksen tausta, tavoite, aineisto, menetelmä ja analyysi,\n"
        "- perustella väitteet lähteillä,\n"
        "- välttää liian yleisiä väitteitä,\n"
        "- käyttää täsmällisiä käsitteitä.\n"
    )

    save_course_material(
        user_id=user_id,
        filename="demokurssimateriaali.txt",
        content=demo_material,
        file_hash="demo-material-v1"
    )

    st.session_state.course_context = load_course_context(user_id)


def reset_demo_user(user_id: str):
    delete_user_data(user_id)
    ensure_user(user_id)
    add_demo_profile(user_id)
    add_demo_course_material(user_id)
    st.session_state.profile = load_profile(user_id)
    st.session_state.course_context = load_course_context(user_id)


# ============================================================
# Otsikko ja kuvaus
# ============================================================

st.title("Demo AI-työkalusta opinnäytetyön tueksi")

st.caption(
    "Pilvipohjainen Streamlit-prototyyppi, jossa työnkulun ohjaaja ja erikoistuneet agentit "
    "tukevat suunnittelua, kirjoittamista, tutkimusasetelmaa, reflektiota ja ohjaukseen valmistautumista."
)

with st.expander("Demo kolmessa vaiheessa", expanded=True):
    st.markdown(
        "**Nopea demopolku:**\n\n"
        "1. Valitse sivupalkista **Täytä demoprofiili** ja **Lisää demokurssimateriaali**.\n"
        "2. Avaa välilehti **Ennakoiva valmentaja** ja paina **Käytä demotilannetta**.\n"
        "3. Paina **Pyydä ennakoivaa valmennusta** ja seuraa, miten agenttinen työnkulku etenee.\n\n"
        "Demon tarkoitus on näyttää, miten eri agentit tarkastelevat opiskelijan tilannetta eri näkökulmista "
        "ja miten lopullinen opiskelijalle suunnattu vastaus koostetaan."
    )

with st.expander("Mikä tämä prototyyppi on?", expanded=False):
    st.markdown(
        "Tämä prototyyppi on **agenttinen opinnäytetyövalmentaja** korkeakouluopiskelijoille.\n\n"
        "Sen tarkoitus on tukea opiskelijaa:\n"
        "- opinnäytetyöprosessin suunnittelussa,\n"
        "- tutkimuskysymyksen ja tutkimusasetelman jäsentämisessä,\n"
        "- tekstiluonnosten formatiivisessa palautteessa,\n"
        "- viikoittaisessa etenemisen seurannassa,\n"
        "- ohjaustapaamisiin valmistautumisessa.\n\n"
        "**Agenttisuus tarkoittaa tässä**, että yksi vastaus muodostuu usean erikoistuneen agentin yhteistyönä.\n\n"
        "**Rajaus:** työkalu ei kirjoita opinnäytetyötä opiskelijan puolesta eikä korvaa ohjaajaa."
    )

with st.expander("Pedagoginen lähtökohta", expanded=False):
    st.markdown(
        "Työkalu tukee opiskelijan **itsesäätelyä, suunnittelua ja ohjaukseen valmistautumista**.\n\n"
        "Se ei pyri korvaamaan ohjaajaa eikä kirjoita opinnäytetyötä opiskelijan puolesta. "
        "Sen tehtävä on auttaa opiskelijaa:\n"
        "- jäsentämään omaa tilannettaan,\n"
        "- tunnistamaan seuraavat konkreettiset askeleet,\n"
        "- muotoilemaan parempia kysymyksiä ohjaajalle,\n"
        "- tarkastelemaan työtään suhteessa annettuihin tavoitteisiin ja kriteereihin."
    )

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


# ============================================================
# Sivupalkki
# ============================================================

st.sidebar.header("Käyttäjä")

st.sidebar.warning(
    "DEMOVERSIO — älä syötä oikeita henkilötietoja, opiskelijanumeroita "
    "tai luottamuksellista tutkimusaineistoa."
)

user_id = st.sidebar.text_input(
    "Pseudonyymi käyttäjätunnus",
    value=st.session_state.get("loaded_user_id", "demo-opiskelija"),
    help=(
        "Käytä esimerkiksi tunnusta 'demo-opiskelija-1'. "
        "Älä käytä oikeaa nimeä, opiskelijanumeroa tai muuta tunnistetta."
    )
).strip()

if not user_id:
    st.sidebar.error("Anna käyttäjätunnus.")
    st.stop()

if st.session_state.get("loaded_user_id") != user_id:
    load_user_state(user_id)

if st.sidebar.button("Lataa käyttäjän tiedot uudelleen"):
    load_user_state(user_id)
    st.sidebar.success("Tiedot ladattu.")

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

if st.sidebar.button("Nollaa demo"):
    reset_demo_user(user_id)
    st.sidebar.success("Demo nollattu ja demotiedot luotu uudelleen.")
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

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "1. Opinnäytetyön profiili",
    "2. Ennakoiva valmentaja",
    "3. Tekstipalaute",
    "4. Tutkimusasetelma",
    "5. Viikkokatsaus",
    "6. Viikkosuunnitelma",
    "7. Ohjaukseen valmistautuminen",
    "8. Ohjaajan kooste",
    "9. Historia"
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
        "Kuvaa tilannettasi, niin Työnkulun ohjaaja valitsee sopivat erikoisagentit "
        "ja muodostaa sinulle seuraavat askeleet."
    )

    if "coach_input_area" not in st.session_state:
        st.session_state.coach_input_area = ""

    if st.button("Käytä demotilannetta", key="demo_coach_input"):
        st.session_state.coach_input_area = (
            "Minulla on aihe ja alustava tutkimuskysymys, mutta en ole varma, "
            "onko tutkimuskysymys liian laaja. Ohjaustapaaminen on ensi viikolla, "
            "ja haluaisin tietää, mitä minun kannattaa valmistella ennen tapaamista. "
            "Tavoitteeni on saada selkeämpi rajaus ja päättää, mitä kysymyksiä esitän ohjaajalle."
        )

    coach_input = st.text_area(
        "Kuvaa tämänhetkinen tilanteesi",
        key="coach_input_area",
        height=180
    )

    if st.button("Pyydä ennakoivaa valmennusta"):
        if not st.session_state.get("profile"):
            st.warning("Tallenna ensin opinnäytetyön profiili.")
        elif not coach_input.strip():
            st.warning("Kuvaa ensin tilanteesi.")
        else:
            with st.spinner("Työnkulun ohjaaja valitsee agentit ja koostaa palautteen. Tämä voi kestää hetken..."):
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

    if "draft_text_area" not in st.session_state:
        st.session_state.draft_text_area = ""

    if st.button("Käytä demotekstiä", key="demo_draft_input"):
        st.session_state.draft_text_area = (
            "Tässä tutkielmassa tarkastelen tekoälyn käyttöä opiskelussa. "
            "Tekoäly on nykyään tärkeä aihe, ja monet opiskelijat käyttävät sitä. "
            "Tutkimukseni selvittää, miten tekoäly vaikuttaa opiskelijoihin. "
            "Aineisto kerätään haastatteluilla ja analysoidaan jotenkin laadullisesti. "
            "Tutkimuksen tavoitteena on ymmärtää opiskelijoiden kokemuksia ja sitä, "
            "miten tekoäly voi auttaa opinnäytetyöprosessissa."
        )

    draft_text = st.text_area(
        "Liitä teksti tähän",
        key="draft_text_area",
        height=320
    )

    if st.button("Anna palautetta tekstistä"):
        if not draft_text.strip():
            st.warning("Liitä ensin tekstiä.")
        else:
            with st.spinner("Tekstiluotsi, Kriteeriluotsi ja Rehellisyysvahti työskentelevät..."):
                run_workflow("draft_feedback", draft_text, user_id, "draft_feedback")


# ============================================================
# Tab 4: tutkimusasetelma
# ============================================================

with tab4:
    st.header("Tutkimusasetelman tuki")

    if "design_input_area" not in st.session_state:
        st.session_state.design_input_area = ""

    if st.button("Käytä demotutkimusasetelmaa", key="demo_design_input"):
        st.session_state.design_input_area = (
            "Tutkimuskysymykseni on: Miten opiskelijat käyttävät tekoälyä opinnäytetyön tekemisessä? "
            "Ajattelen kerätä aineiston 5–6 opiskelijan puolistrukturoiduilla haastatteluilla. "
            "Menetelmänä olisi laadullinen haastattelututkimus ja analyysitapana temaattinen analyysi. "
            "En ole vielä varma, pitäisikö aihe rajata opinnäytetyön suunnitteluun, tekstipalautteen hyödyntämiseen "
            "vai koko opinnäytetyöprosessiin. Haluaisin arvioida, onko tutkimuskysymys liian laaja ja "
            "onko aineisto suhteessa tavoitteeseen riittävä."
        )

    design_input = st.text_area(
        "Kuvaa tutkimuskysymys, aineisto, menetelmä ja suunniteltu analyysi",
        key="design_input_area",
        height=280
    )

    if st.button("Analysoi tutkimusasetelma"):
        if not design_input.strip():
            st.warning("Kuvaa ensin tutkimusasetelmasi.")
        else:
            with st.spinner("Tutkimusluotsi ja Kriteeriluotsi työskentelevät..."):
                run_workflow("research_design", design_input, user_id, "research_design")


# ============================================================
# Tab 5: viikkokatsaus
# ============================================================

with tab5:
    st.header("Viikoittainen tilannekatsaus")

    if "weekly_completed" not in st.session_state:
        st.session_state.weekly_completed = ""

    if "weekly_blocked" not in st.session_state:
        st.session_state.weekly_blocked = ""

    if "weekly_next_action" not in st.session_state:
        st.session_state.weekly_next_action = ""

    if "weekly_support_needed" not in st.session_state:
        st.session_state.weekly_support_needed = ""

    if st.button("Täytä demoviikkokatsaus", key="demo_weekly_checkin"):
        st.session_state.weekly_completed = (
            "Tarkensin aihetta ja luin kolme aiheeseen liittyvää artikkelia. "
            "Kirjoitin myös alustavan version tutkimuksen taustasta."
        )
        st.session_state.weekly_blocked = (
            "En ole varma, miten rajaan tutkimuskysymyksen riittävän kapeaksi. "
            "Lisäksi en tiedä, pitäisikö haastatteluissa keskittyä koko opinnäytetyöprosessiin "
            "vai vain kirjoittamisen tukeen."
        )
        st.session_state.weekly_next_action = (
            "Haluan laatia kaksi vaihtoehtoista tutkimuskysymystä ja valmistella ne ohjaustapaamiseen."
        )
        st.session_state.weekly_support_needed = (
            "Tarvitsen apua rajauksen, seuraavien konkreettisten tehtävien ja ohjaajalle esitettävien kysymysten määrittelyssä."
        )

    completed = st.text_area(
        "Mitä sait tällä viikolla valmiiksi?",
        key="weekly_completed",
        height=100
    )

    blocked = st.text_area(
        "Mikä estää etenemistä?",
        key="weekly_blocked",
        height=100
    )

    next_action = st.text_area(
        "Mitä aiot tehdä seuraavaksi?",
        key="weekly_next_action",
        height=100
    )

    support_needed = st.text_area(
        "Millaista tukea tarvitset?",
        key="weekly_support_needed",
        height=100
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

            with st.spinner("Etenemisluotsi ja Reflektiokumppani työskentelevät..."):
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

    if "weekly_input_area" not in st.session_state:
        st.session_state.weekly_input_area = ""

    if st.button("Käytä demopyyntöä viikkosuunnitelmaan", key="demo_weekly_plan"):
        st.session_state.weekly_input_area = (
            "Minulla on tällä viikolla noin 8 tuntia aikaa. "
            "Haluan valmistella ohjaustapaamista varten tutkimuskysymyksen rajauksen, "
            "kaksi vaihtoehtoista tutkimuskysymystä ja alustavan menetelmäkuvauksen. "
            "Tarvitsen suunnitelman, jossa työ jakautuu realistisesti useammalle päivälle."
        )

    weekly_input = st.text_area(
        "Valinnainen tarkennus suunnitelmalle",
        key="weekly_input_area",
        height=160
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

            with st.spinner("Viikkovalmentaja työskentelee..."):
                run_workflow("weekly_plan", user_input, user_id, "weekly_plan")


# ============================================================
# Tab 7: ohjaukseen valmistautuminen
# ============================================================

with tab7:
    st.header("Ohjaukseen valmistautuminen")

    st.write(
        "Luo tiivis muistio ohjaustapaamista varten. Tarkista ja muokkaa muistio ennen jakamista."
    )

    if "supervision_input_area" not in st.session_state:
        st.session_state.supervision_input_area = ""

    if st.button("Käytä demo-ohjaustilannetta", key="demo_supervision_input"):
        st.session_state.supervision_input_area = (
            "Haluan keskustella ohjaajan kanssa siitä, onko tutkimuskysymykseni liian laaja, "
            "riittääkö 5–6 haastattelua aineistoksi ja kannattaako analyysitavaksi valita temaattinen analyysi. "
            "Lisäksi tarvitsen päätöksen siitä, pitäisikö tutkimuksen keskittyä tekoälyn käyttöön koko opinnäytetyöprosessissa "
            "vai rajatummin kirjoittamisen suunnitteluun ja tekstipalautteeseen. "
            "Haluan lähteä tapaamisesta selkeän seuraavan viikon tehtävälistan kanssa."
        )

    supervision_input = st.text_area(
        "Kuvaa, mitä haluat käsitellä ohjaajan kanssa",
        key="supervision_input_area",
        height=180
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

            with st.spinner("Ohjaustapaamisen valmistelija työskentelee..."):
                run_workflow(
                    "supervision_summary",
                    user_input,
                    user_id,
                    "supervision_summary"
                )


# ============================================================
# Tab 8: ohjaajan kooste
# ============================================================

with tab8:
    st.header("Ohjaajan kooste")

    st.write(
        "Tämä näkymä tuottaa opiskelijan itse tarkistettavan koosteen ohjaajalle. "
        "Koostetta ei tule jakaa automaattisesti, vaan opiskelijan tulee tarkistaa ja muokata se ensin."
    )

    if "supervisor_summary_area" not in st.session_state:
        st.session_state.supervisor_summary_area = ""

    if st.button("Käytä demopyyntöä ohjaajan koosteeseen", key="demo_supervisor_summary"):
        st.session_state.supervisor_summary_area = (
            "Haluan tiivistää ohjaajalle nykyisen tilanteeni, tutkimuskysymyksen rajauksen ongelman, "
            "aineiston riittävyyteen liittyvän epävarmuuden sekä päätökset, joita tarvitsen seuraavaksi."
        )

    supervisor_summary_input = st.text_area(
        "Mitä haluat nostaa ohjaajalle?",
        key="supervisor_summary_area",
        height=160
    )

    if st.button("Luo ohjaajan kooste"):
        if not st.session_state.get("profile"):
            st.warning("Tallenna ensin opinnäytetyön profiili.")
        else:
            if supervisor_summary_input.strip():
                user_input = supervisor_summary_input.strip()
            else:
                user_input = (
                    "Laadi opiskelijan tarkistettava kooste ohjaajalle. "
                    "Keskity nykyiseen vaiheeseen, etenemiseen, esteisiin, tarvittaviin päätöksiin "
                    "ja kysymyksiin ohjaajalle. Älä sisällytä tarpeettomia henkilötietoja."
                )

            with st.spinner("Ohjaajan koostetta muodostetaan..."):
                run_workflow(
                    "supervision_summary",
                    user_input,
                    user_id,
                    "supervisor_summary"
                )


# ============================================================
# Tab 9: historia
# ============================================================

with tab9:
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
