"""Spanish requests reach domain tools through keyword routes.

The tool index embeds English descriptions with MiniLM, so Spanish requests
score like unrelated tools; ``_classify_agent_request`` must route them.
"""
import pytest

from src.agent_loop import _classify_agent_request


def _domains(text):
    result = _classify_agent_request([{"role": "user", "content": text}], text)
    return result["domains"], result["low_signal"]


@pytest.mark.parametrize(
    "text,domain",
    [
        ("recuérdame mañana a las 9 llamar al dentista", "notes_calendar_tasks"),
        ("apunta en una nota la lista de la compra", "notes_calendar_tasks"),
        ("añade una reunión a mi calendario el viernes", "notes_calendar_tasks"),
        ("cada mañana resume mis correos", "notes_calendar_tasks"),
        ("revisa mi bandeja de entrada por correos de Ana", "email"),
        ("redacta un correo para el casero", "email"),
        ("busca en internet el precio de la RX 9070", "web"),
        ("¿qué tiempo hace hoy en Madrid?", "web"),
        ("escribe un poema sobre el mar", "documents"),
        ("lee el archivo config.yaml de la carpeta del proyecto", "files"),
        ("guarda el teléfono de Luis en mis contactos", "contacts"),
        ("¿qué modelos tengo instalados?", "cookbook"),
    ],
)
def test_spanish_requests_select_domain(text, domain):
    domains, low_signal = _domains(text)
    assert domain in domains
    assert not low_signal


def test_spanish_reminder_is_not_routed_to_documents():
    domains, _ = _domains("escribe un recordatorio para el lunes")
    assert "notes_calendar_tasks" in domains
    assert "documents" not in domains


def test_plain_spanish_chat_stays_low_signal():
    domains, low_signal = _domains("¿por qué el cielo es azul?")
    assert domains == set()
    assert low_signal
