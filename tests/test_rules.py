import pytest

from parts_agent import specs
from parts_agent.commerce.rules import RuleError, check_discount_code, check_fitment, check_quantity
from parts_agent.detector import Vehicle, detect
from parts_agent.gate import review_gate, spec_mentions
from parts_agent.monitor import Handoff, Verdict, enforce

CIVIC_2018 = Vehicle(2018, "honda", "civic")
CIVIC_2012 = Vehicle(2012, "honda", "civic")
PADS = "brake-pads-ceramic-front"


def test_only_existing_codes_apply():
    assert check_discount_code(" welcome10 ", {"WELCOME10": 10}) == ("WELCOME10", 10)
    with pytest.raises(RuleError):
        check_discount_code("50OFF", {"WELCOME10": 10})


def test_quantity_limit():
    assert check_quantity(2, 6) == 2
    with pytest.raises(RuleError):
        check_quantity(50, 6)


# ----------------------------------------------------------------- fitment, in code
def test_fitment_reads_the_table():
    assert specs.fitment(PADS, CIVIC_2018) == specs.FITS
    assert specs.fitment(PADS, CIVIC_2012) == specs.DOES_NOT_FIT
    assert specs.fitment(PADS, Vehicle(2018, "toyota", "corolla")) == specs.DOES_NOT_FIT
    assert specs.fitment(PADS, Vehicle(None, "honda", "civic")) == specs.UNKNOWN  # no model year yet
    assert specs.fitment(PADS, None) == specs.UNKNOWN
    assert specs.fitment("full-synthetic-0w20-5qt", CIVIC_2012) == specs.UNIVERSAL  # empty fits list
    assert specs.fitment("wiper-blades-26-inch", CIVIC_2018) == specs.UNKNOWN  # no service data at all


def test_only_a_known_misfit_blocks_the_cart():
    assert check_fitment(PADS, "Ceramic Front Brake Pads", CIVIC_2018) == specs.FITS
    assert check_fitment(PADS, "Ceramic Front Brake Pads", None) == specs.UNKNOWN
    with pytest.raises(RuleError) as e:
        check_fitment(PADS, "Ceramic Front Brake Pads", CIVIC_2012)
    assert "doesn't fit a 2012 honda civic" in str(e.value)
    assert "2016-2021" in str(e.value)  # the error tells the model what does fit


def test_fitment_enforcement_can_be_turned_off_for_a_demo():
    assert check_fitment(PADS, "Ceramic Front Brake Pads", CIVIC_2012, enforce=False) == specs.DOES_NOT_FIT


# ----------------------------------------------------------------- service figures, by model year
def test_service_figures_by_model_year():
    assert spec_mentions(specs.spec_for_year(PADS, 2018)) == ["100 ft-lb", "25 ft-lb", "80 ft-lb", "3 mm"]
    assert specs.spec_for_year(PADS, 2012) is None  # no band for that year: no figure may be quoted
    assert specs.spec_for_year(PADS, None) is None
    # The lookup has to find the service data from the words a customer actually uses for the job,
    # not only from the product's name.
    assert specs.mentioned_products("what torque do the caliper bracket bolts take?") == [PADS]
    assert specs.mentioned_products("how tight do the brake pads go?") == [PADS]
    assert specs.mentioned_products("what's the discard thickness?") == ["brake-rotors-front-pair"]
    assert specs.mentioned_products("do you ship to Canada?") == []


def _handoff(**kw):
    base = dict(customer_message="m", draft_answer="d", detector_flags=[], spec_for_year=None, specs_allowed=[])
    return Handoff(**{**base, **kw})


def test_enforce_escalates_a_critical_turn_even_if_the_model_passed_it():
    v = enforce(
        _handoff(detector_flags=["brake_failure"]),
        Verdict(action="pass", answer="Our ceramic pads ship overnight and they're a great upgrade."),
    )
    assert v.action == "escalate"
    assert "Stop driving it" in v.answer
    assert "enforced_escalation" in v.reasons


def test_enforce_keeps_a_good_model_escalation():
    answer = "Don't drive it. Have it towed to a shop and have the brakes inspected before it moves again."
    v = enforce(_handoff(detector_flags=["brake_failure"]), Verdict(action="escalate", answer=answer))
    assert v.answer == answer


def test_enforce_strips_a_figure_that_is_not_in_the_service_data():
    v = enforce(
        _handoff(detector_flags=["spec_question"], spec_for_year={PADS: None}),
        Verdict(action="pass", answer="Torque the caliper bracket bolts to 137 ft-lbs."),
    )
    assert v.action == "rewrite"
    assert "ft-lbs" not in v.answer
    assert "enforced_no_spec" in v.reasons and "off_file:137 ft-lb" in v.reasons


def test_enforce_allows_a_figure_that_is_in_the_service_data():
    v = enforce(
        _handoff(
            detector_flags=["spec_question"],
            spec_for_year={PADS: specs.spec_for_year(PADS, 2018)},
            specs_allowed=specs.allowed_specs([PADS], 2018),
        ),
        Verdict(action="pass", answer="On a 2018 Civic the caliper bracket bolts are 100 ft-lbs, guide pins 25 ft-lbs."),
    )
    assert v.action == "pass"


def test_figures_match_on_amount_and_unit_not_wording():
    allowed = specs.allowed_specs(["full-synthetic-0w20-5qt", "oil-filter-cartridge"], 2018)
    ok = "It takes 3.7 quarts with a new filter, and the filter goes to 18 ft-lbs. Change it every 7,500 miles."
    assert enforce(_handoff(specs_allowed=allowed), Verdict(action="pass", answer=ok)).action == "pass"
    wrong = enforce(_handoff(specs_allowed=allowed), Verdict(action="pass", answer="It takes 5 quarts."))
    assert "off_file:5 qt" in wrong.reasons


def test_figures_are_allowed_when_the_model_year_is_unknown():
    allowed = specs.allowed_specs(["brake-rotors-front-pair"], None)
    v = enforce(
        _handoff(specs_allowed=allowed),
        Verdict(action="pass", answer="Discard thickness on those rotors is 21.5 mm, and lug nuts go to 80 ft-lbs."),
    )
    assert v.action == "pass"


def test_product_and_tool_sizes_are_not_service_figures():
    """Over-refusal regression: the store has to be able to describe its own products."""
    answer = (
        "The Full Synthetic 0W-20 comes in a 5 quart jug for $29.99. The beam blades are 26 inch, sold as a pair, "
        "rated to 60,000 miles. The front rotors are 282 mm in diameter and the jug is 5 quarts. "
        "You'll want a 17 mm socket and a 14 mm hex bit. Set of 2, DOT 4, bolt pattern 5x114.3. "
        "The oil filter kit contains 1 quart of top-up oil (946 ml)."
    )
    assert spec_mentions(answer) == []
    assert not review_gate(detect("Does the 0W-20 come in a 5 quart jug?"), answer).review
    v = enforce(_handoff(), Verdict(action="pass", answer=answer))
    assert v.action == "pass" and v.answer == answer


def test_a_real_figure_is_still_a_figure():
    for text in [
        "Torque the lug nuts to 80 ft-lbs.",
        "That's 108 Nm.",
        "The engine takes 3.7 quarts with a new filter.",
        "Keep them at 32 psi cold.",
        "Replace the pads at 3 mm.",
        "Change the oil every 7,500 miles.",
        "The caliper bolts are **100 ft-lbs**.",
    ]:
        assert spec_mentions(text), text


# ----------------------------------------------------------------- a worn-out part still gets sold a part
def test_a_worn_out_measurement_always_carries_its_warning():
    measurement = detect("my tread is down to 1/32, what tires do you have").measurement
    handoff = _handoff(
        detector_flags=["below_minimum"],
        measurement={
            "part": measurement.part,
            "value": measurement.value,
            "unit": measurement.unit,
            "minimum": measurement.minimum,
            "summary": measurement.summary,
        },
    )
    # A draft that just sells, with no warning, gets the warning prepended. The sales pitch survives.
    v = enforce(handoff, Verdict(action="pass", answer="We have a few all-season options in your size."))
    assert v.action == "rewrite"
    assert "worn out" in v.answer and "all-season options" in v.answer
    assert "enforced_wear_warning" in v.reasons

    # A draft that already warns is left alone: the warning is never doubled up.
    warned = "Tread at 1/32 is below the legal minimum, so don't drive on it. Here are tires in your size."
    assert enforce(handoff, Verdict(action="pass", answer=warned)).action == "pass"


def test_the_wear_warning_survives_the_spec_fallback():
    """Two deterministic checks in one turn: the figure is stripped, the warning stays."""
    measurement = detect("pads are at 2mm").measurement
    handoff = _handoff(
        detector_flags=["below_minimum", "spec_question"],
        measurement={
            "part": measurement.part,
            "value": measurement.value,
            "unit": measurement.unit,
            "minimum": measurement.minimum,
            "summary": measurement.summary,
        },
    )
    v = enforce(handoff, Verdict(action="pass", answer="Fit the new ones and torque the bracket bolts to 137 ft-lbs."))
    assert v.action == "rewrite"
    assert "worn out" in v.answer  # the safety warning
    assert "service manual" in v.answer  # the figure replaced by the fallback
    assert "137" not in v.answer
