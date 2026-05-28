import os
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# Ympäristöasetukset
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH, override=True)


def get_api_key() -> Optional[str]:
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        return api_key

    try:
        import streamlit as st
        api_key = st.secrets.get("OPENAI_API_KEY", None)
        if api_key:
            return api_key
    except Exception:
        pass

    return None


OPENAI_API_KEY = get_api_key()

FAST_MODEL = os.getenv("OPENAI_FAST_MODEL", "gpt-4.1-nano")
STRONG_MODEL = os.getenv("OPENAI_STRONG_MODEL", "gpt-4o-mini")
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", STRONG_MODEL)

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ============================================================
# Yleiset apufunktiot
# ============================================================

def trunc(text: str, n: int = 12000) -> str:
    if not text:
        return ""
    return text[:n]


def safe_json(data: Any) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:
        return str(data)


def extract_json_object(text: str) -> Dict[str, Any]:
    """
    Yrittää poimia JSON-olion LLM-vastauksesta.
    Jos jäsennys epäonnistuu, palauttaa tyhjän sanakirjan.
    """

    if not text:
        return {}

    cleaned = text.strip()
    cleaned = cleaned.replace("```json", "")
    cleaned = cleaned.replace("```", "")
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)

    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return {}

    return {}


def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float = 0.3
) -> str:
    if client is None:
        return (
            "OpenAI API -avain puuttuu. Lisää OPENAI_API_KEY .env-tiedostoon "
            "tai Streamlitin secrets-asetuksiin."
        )

    language_instruction = (
        "\n\nKieliohje:\n"
        "- Vastaa aina suomeksi.\n"
        "- Käytä selkeää, akateemista ja opiskelijaa tukevaa suomen kieltä.\n"
        "- Ole konkreettinen ja käytännöllinen.\n"
        "- Älä kirjoita opiskelijan opinnäytetyötä hänen puolestaan.\n"
        "- Älä keksi lähteitä, aineistoja, tuloksia, eettisiä lupia tai organisaation sääntöjä.\n"
        "- Jos tieto puuttuu, sano se selvästi.\n"
        "- Muistuta tarvittaessa, että opiskelijan tulee varmistaa ratkaisut ohjaajalta."
    )

    try:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt + language_instruction
                },
                {
                    "role": "user",
                    "content": user_prompt
                },
            ],
        )

        return response.choices[0].message.content

    except Exception as e:
        return "LLM-kutsu epäonnistui: " + str(e)


def agent_outputs_text(outputs: Dict[str, str]) -> str:
    if not outputs:
        return "Ei aiempia agenttien tuotoksia."

    parts = []

    for name, output in outputs.items():
        parts.append(
            "\n\n--- AGENTIN "
            + str(name).upper()
            + " TUOTOS ---\n"
            + str(output)
        )

    return "\n".join(parts)


def model_for_agent(agent_name: str, override: Optional[str] = None) -> str:
    if override:
        return override

    config = AGENTS.get(agent_name, {})
    tier = config.get("tier", "strong")

    if tier == "fast":
        return FAST_MODEL

    if tier == "strong":
        return STRONG_MODEL

    return DEFAULT_MODEL


# ============================================================
# Agenttimääritykset
# ============================================================

AGENTS = {
    "integrity": {
        "label": "akateeminen rehellisyys",
        "tier": "fast",
        "temperature": 0.1,
        "system": (
            "Olet akateemisen rehellisyyden ja turvallisuuden agentti. "
            "Tehtäväsi on arvioida, liittyykö opiskelijan pyyntöön tai tekstiin riskejä: "
            "haamukirjoittaminen, plagiointi, keksityt lähteet, keksitty aineisto, "
            "epäeettinen tutkimus, liian vahvat väitteet, ohjaajan päätöksenteon korvaaminen "
            "tai luottamuksellisen tiedon käsittely. "
            "Älä ole tarpeettoman varoittava, mutta merkitse riskit selkeästi. "
            "Anna: riskitaso, keskeinen huoli, turvallinen tukitapa ja lyhyt rajausmuistutus."
        ),
    },
    "planner": {
        "label": "suunnittelu",
        "tier": "fast",
        "temperature": 0.3,
        "system": (
            "Olet opinnäytetyön suunnitteluagentti. "
            "Auta opiskelijaa hahmottamaan nykyinen vaihe, seuraavat konkreettiset askeleet, "
            "realistinen aikataulu, mahdolliset riskit ja ohjaajalle esitettävät kysymykset. "
            "Tavoitteena on tukea opiskelijan omaa toimijuutta ja etenemistä."
        ),
    },
    "writing_coach": {
        "label": "kirjoituspalaute",
        "tier": "strong",
        "temperature": 0.3,
        "system": (
            "Olet akateemisen kirjoittamisen palauteagentti. "
            "Anna formatiivista palautetta tekstin selkeydestä, rakenteesta, johdonmukaisuudesta, "
            "argumentaatiosta, akateemisesta tyylistä, käsitteiden käytöstä ja evidenssin tarpeesta. "
            "Älä kirjoita kokonaisia kappaleita opiskelijan puolesta. "
            "Voit antaa lyhyitä havainnollistavia esimerkkejä, mutta opiskelijan tulee säilyä tekstin tekijänä."
        ),
    },
    "criteria_alignment": {
        "label": "kriteerivastaavuus",
        "tier": "strong",
        "temperature": 0.2,
        "system": (
            "Olet kriteerien ja ohjeiden vastaavuuden arvioija. "
            "Vertaa opiskelijan suunnitelmaa, tekstiä tai tilannetta annettuihin kurssimateriaaleihin, "
            "opinnäytetyöohjeisiin, arviointikriteereihin ja tavoitteisiin. "
            "Älä keksi organisaation sääntöjä. "
            "Jos kurssimateriaalia ei ole riittävästi, kerro arvioinnin rajallisuus selvästi."
        ),
    },
    "research_design": {
        "label": "tutkimusasetelma",
        "tier": "strong",
        "temperature": 0.3,
        "system": (
            "Olet tutkimusasetelman tukija. "
            "Arvioi tutkimuskysymyksen selkeyttä, aiheen rajausta, aineiston tai materiaalin sopivuutta, "
            "menetelmän ja analyysitavan yhteensopivuutta, toteutettavuutta ja mahdollisia eettisiä kysymyksiä. "
            "Älä tee lopullisia menetelmäpäätöksiä opiskelijan puolesta. "
            "Nosta esiin asiat, jotka kannattaa varmistaa ohjaajalta."
        ),
    },
    "reflection": {
        "label": "reflektio ja itsesäätely",
        "tier": "fast",
        "temperature": 0.4,
        "system": (
            "Olet reflektio- ja itsesäätelyagentti. "
            "Tue opiskelijaa oman etenemisen arvioinnissa, esteiden tunnistamisessa, "
            "realistisessa suunnittelussa, motivaation ylläpitämisessä ja seuraavan pienen askeleen valinnassa. "
            "Ole käytännöllinen, rauhallinen ja tiivis."
        ),
    },
    "weekly_plan": {
        "label": "viikkosuunnitelma",
        "tier": "fast",
        "temperature": 0.3,
        "system": (
            "Olet viikkosuunnitelma-agentti. "
            "Laadi realistinen seitsemän päivän suunnitelma opinnäytetyön etenemiseksi. "
            "Suunnitelman tulee sisältää konkreettiset tehtävät, vähimmäistavoite, riskit ja mahdolliset ohjaajakysymykset. "
            "Älä kirjoita opinnäytetyön sisältöä opiskelijan puolesta."
        ),
    },
    "supervision_summary": {
        "label": "ohjausmuistio",
        "tier": "fast",
        "temperature": 0.3,
        "system": (
            "Olet ohjaukseen valmistautumisen agentti. "
            "Laadi opiskelijalle tiivis ja muokattava muistio ohjaustapaamista varten. "
            "Keskity tilanteeseen, etenemiseen, esteisiin, tarvittaviin päätöksiin, ohjaajalle esittäviin kysymyksiin "
            "ja seuraaviin askeleisiin. "
            "Älä arvioi opiskelijaa summatiivisesti."
        ),
    },
}


# ============================================================
# Orkestroija ja koostaja
# ============================================================

ORCHESTRATOR_SYSTEM = (
    "Olet agenttisen opinnäytetyövalmentajan orkestroija-agentti. "
    "Valitse, mitkä erikoistuneet agentit tarvitaan opiskelijan pyyntöön. "
    "Palauta vain validia JSONia. JSON-avainten tulee olla englanniksi, mutta tekstiarvojen suomeksi. "
    "Käytettävissä olevat agentit ovat: integrity, planner, writing_coach, criteria_alignment, "
    "research_design, reflection, weekly_plan, supervision_summary. "
    "Valitse vain tarpeelliset agentit, jotta vastaus pysyy demossa nopeana. "
    "Sisällytä integrity, kun pyyntö liittyy kirjoittamiseen, tutkimukseen, aineistoon, lähteisiin tai opinnäytetyön sisältöön. "
    "Palauta täsmälleen tämä rakenne: "
    "{\"task_analysis\":\"lyhyt analyysi\", "
    "\"agents\":[\"integrity\",\"planner\"], "
    "\"reason\":\"miksi nämä agentit valittiin\", "
    "\"expected_final_response\":\"mitä lopullisen vastauksen tulisi tarjota\"}"
)


FINALIZER_SYSTEM = (
    "Olet lopullisen vastauksen koostaja agenttisessa opinnäytetyövalmentajassa. "
    "Koosta erikoisagenttien tuotoksista yksi selkeä, opiskelijalle suunnattu vastaus. "
    "Älä liitä agenttien tuotoksia sellaisenaan. Poista toisto ja ristiriidat. "
    "Pidä vastaus käytännöllisenä ja pedagogisesti hyödyllisenä. "
    "Älä kirjoita opinnäytetyötä opiskelijan puolesta. "
    "Älä keksi lähteitä, aineistoja, tuloksia tai organisaation sääntöjä."
)


# ============================================================
# Promptin rakentaminen
# ============================================================

def build_agent_prompt(agent_name: str, state: Dict[str, Any]) -> str:
    return (
        "Agentti:\n"
        + agent_name
        + "\n\nTehtävätyyppi:\n"
        + str(state.get("task_type", ""))
        + "\n\nOpiskelijan profiili:\n"
        + safe_json(state.get("profile", {}))
        + "\n\nOpiskelijan syöte:\n"
        + trunc(state.get("user_input", ""), 16000)
        + "\n\nKurssikonteksti tai haetut materiaalikatkelmat:\n"
        + trunc(state.get("course_context", ""), 14000)
        + "\n\nViimeaikainen keskusteluhistoria:\n"
        + trunc(state.get("recent_history", ""), 6000)
        + "\n\nViimeaikaiset viikkokatsaukset:\n"
        + safe_json(state.get("checkins", []))
        + "\n\nAiemmat agenttituotokset tässä työnkulussa:\n"
        + agent_outputs_text(state.get("agent_outputs", {}))
        + "\n\nVastaa oman agenttiroolisi mukaisesti. "
        + "Ole konkreettinen, tiivis ja pedagogisesti hyödyllinen."
    )


def build_finalizer_prompt(state: Dict[str, Any], route: Dict[str, Any]) -> str:
    return (
        "Orkestroijan reitti:\n"
        + safe_json(route)
        + "\n\nTehtävätyyppi:\n"
        + str(state.get("task_type", ""))
        + "\n\nOpiskelijan profiili:\n"
        + safe_json(state.get("profile", {}))
        + "\n\nOpiskelijan syöte:\n"
        + trunc(state.get("user_input", ""), 10000)
        + "\n\nAgenttien tuotokset:\n"
        + agent_outputs_text(state.get("agent_outputs", {}))
        + "\n\nLaadi lopullinen opiskelijalle suunnattu vastaus seuraavalla rakenteella:\n\n"
        + "## Pääsuositus\n\n"
        + "## Keskeinen palaute\n\n"
        + "## Seuraavat askeleet\n\n"
        + "## Riskit ja huomioitavat asiat\n\n"
        + "## Kysymykset ohjaajalle\n\n"
        + "## Reflektiokysymys\n\n"
        + "Pidä vastaus demotilanteeseen sopivan selkeänä ja kohtuullisen tiiviinä."
    )


# ============================================================
# Reititys
# ============================================================

def fallback_agents_for_task(task_type: str) -> List[str]:
    """
    Demoversioon optimoidut oletusreitit.
    Nämä pitävät vasteajan kohtuullisena mutta säilyttävät agenttisuuden.
    """

    if task_type == "proactive_coach":
        return ["integrity", "planner", "criteria_alignment", "reflection"]

    if task_type == "draft_feedback":
        return ["integrity", "writing_coach", "criteria_alignment"]

    if task_type == "research_design":
        return ["integrity", "research_design", "criteria_alignment"]

    if task_type == "weekly_checkin":
        return ["planner", "reflection"]

    if task_type == "weekly_plan":
        return ["weekly_plan", "reflection"]

    if task_type == "supervision_summary":
        return ["supervision_summary", "planner"]

    return ["integrity", "planner"]


def max_agents_for_task(task_type: str) -> int:
    """
    Rajaa agenttien määrää demossa, jotta vasteajat eivät kasva liikaa.
    """

    if task_type in ["draft_feedback", "research_design"]:
        return 3

    if task_type in ["weekly_plan", "supervision_summary", "weekly_checkin"]:
        return 2

    return 4


def normalize_agents(agent_names: List[str], task_type: str) -> List[str]:
    if not agent_names:
        agent_names = fallback_agents_for_task(task_type)

    cleaned = []

    for name in agent_names:
        if name in AGENTS and name not in cleaned:
            cleaned.append(name)

    if not cleaned:
        cleaned = fallback_agents_for_task(task_type)

    # Akateemisen rehellisyyden agentti mukaan tilanteisiin, joissa käsitellään sisältöä.
    needs_integrity = task_type in [
        "proactive_coach",
        "draft_feedback",
        "research_design"
    ]

    if needs_integrity and "integrity" not in cleaned:
        cleaned.insert(0, "integrity")

    limit = max_agents_for_task(task_type)
    cleaned = cleaned[:limit]

    # Varmistetaan vielä, ettei integrity putoa pois tärkeissä tehtävissä.
    if needs_integrity and "integrity" not in cleaned:
        cleaned = ["integrity"] + cleaned
        cleaned = cleaned[:limit]

    return cleaned


# ============================================================
# Päätyönkulku
# ============================================================

def run_agentic_workflow(
    task_type: str,
    user_input: str,
    profile: Dict[str, Any],
    course_context: str = "",
    recent_history: str = "",
    checkins: Optional[List[Dict[str, Any]]] = None,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Ajaa agenttisen työnkulun.

    Palauttaa:
    {
        "route": dict,
        "selected_agents": list,
        "agent_outputs": dict,
        "final_response": str
    }
    """

    if checkins is None:
        checkins = []

    state = {
        "task_type": task_type,
        "user_input": user_input or "",
        "profile": profile or {},
        "course_context": course_context or "",
        "recent_history": recent_history or "",
        "checkins": checkins,
        "agent_outputs": {},
    }

    # 1. Orkestroija
    orchestrator_prompt = (
        "Tehtävätyyppi:\n"
        + str(task_type)
        + "\n\nOpiskelijan syöte:\n"
        + trunc(user_input, 10000)
        + "\n\nOpiskelijan profiili:\n"
        + safe_json(profile or {})
        + "\n\nKurssikontekstia saatavilla:\n"
        + ("kyllä" if course_context else "ei")
        + "\n\nViimeaikaisia viikkokatsauksia:\n"
        + ("kyllä" if checkins else "ei")
        + "\n\nValitse tarvittavat agentit. Pidä reititys demotilanteessa tehokkaana."
    )

    route_raw = call_llm(
        ORCHESTRATOR_SYSTEM,
        orchestrator_prompt,
        model_for_agent("integrity", model),
        temperature=0.1
    )

    route = extract_json_object(route_raw)

    if not route:
        route = {
            "task_analysis": "Orkestroijan JSON-vastausta ei voitu jäsentää. Käytetään oletusreititystä.",
            "agents": fallback_agents_for_task(task_type),
            "reason": "Oletusreititys tehtävätyypin perusteella.",
            "expected_final_response": "Anna opiskelijalle käytännöllistä opinnäytetyön tukea."
        }

    selected_agents = normalize_agents(route.get("agents", []), task_type)

    # 2. Erikoisagentit
    for agent_name in selected_agents:
        config = AGENTS[agent_name]
        prompt = build_agent_prompt(agent_name, state)

        output = call_llm(
            config["system"],
            prompt,
            model_for_agent(agent_name, model),
            temperature=config.get("temperature", 0.3)
        )

        state["agent_outputs"][agent_name] = output

    # 3. Lopullinen koostaja
    final_prompt = build_finalizer_prompt(state, route)

    final_response = call_llm(
        FINALIZER_SYSTEM,
        final_prompt,
        model_for_agent("writing_coach", model),
        temperature=0.3
    )

    return {
        "route": route,
        "selected_agents": selected_agents,
        "agent_outputs": state["agent_outputs"],
        "final_response": final_response
    }


# ============================================================
# Debug-näkymä Streamlitiin
# ============================================================

def format_workflow_debug(result: Dict[str, Any]) -> str:
    selected_agents = result.get("selected_agents", [])
    labels = []

    for name in selected_agents:
        label = AGENTS.get(name, {}).get("label", name)
        labels.append(label + " (" + name + ")")

    return (
        "## Malliasetukset\n\n"
        + "- Nopea malli: `" + str(FAST_MODEL) + "`\n"
        + "- Vahva malli: `" + str(STRONG_MODEL) + "`\n"
        + "- Oletusmalli: `" + str(DEFAULT_MODEL) + "`\n\n"
        + "## Orkestroijan reitti\n\n"
        + "```json\n"
        + safe_json(result.get("route", {}))
        + "\n```\n\n"
        + "## Valitut agentit\n\n"
        + ", ".join(labels)
        + "\n\n"
        + "## Agenttien tuotokset\n\n"
        + agent_outputs_text(result.get("agent_outputs", {}))
    )
