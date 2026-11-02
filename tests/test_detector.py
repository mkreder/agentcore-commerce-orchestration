import pytest

from parts_agent.detector import Vehicle, detect
from parts_agent.gate import review_gate, stream_mode


@pytest.mark.parametrize(
    "message",
    [
        "My brake pedal goes almost to the floor and it's grinding. Which front pads ship fastest?",
        "pedal went straight to the floor this morning, do you sell brake fluid?",
        "metal on metal grinding every time I brake",
        "I lost all my brake fluid, there's a puddle",
        "the brakes don't work at all now",
    ],
)
def test_brake_failure_is_flagged(message):
    assert "brake_failure" in detect(message).flags


@pytest.mark.parametrize(
    ("message", "flag"),
    [
        ("the wheel came off while I was driving", "loss_of_control"),
        ("my front wheel is wobbling and the lug nuts might be loose", "loss_of_control"),
        ("steering went loose and the car wanders all over the lane", "loss_of_control"),
        ("the front spring snapped and the tire is rubbing the fender", "loss_of_control"),
        ("I think the ball joint separated in the parking lot", "loss_of_control"),
        ("I smell gasoline in the cabin and under the hood", "fire_or_fuel"),
        ("there's fuel dripping off the fuel rail", "fire_or_fuel"),
        ("smoke is coming from under my hood", "fire_or_fuel"),
        ("my battery is swollen and hissing", "fire_or_fuel"),
        ("temp gauge is in the red and steam is pouring out", "fire_or_fuel"),
        ("what torque do the caliper bracket bolts take?", "spec_question"),
        ("how many quarts of oil does it take?", "spec_question"),
        ("what's the oil change interval?", "spec_question"),
        ("could it be the wheel bearing?", "diagnosis_request"),
        ("why does my car shake at 60?", "diagnosis_request"),
    ],
)
def test_other_flags(message, flag):
    assert flag in detect(message).flags


def test_the_vehicle_is_read_from_the_message():
    d = detect("Do you have ceramic front brake pads for a 2018 Honda Civic?")
    assert d.vehicle == Vehicle(year=2018, make="honda", model="civic")
    assert d.flags == ()


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("my 2018 honda civic needs pads", Vehicle(2018, "honda", "civic")),
        ("Honda Civic, 2016", Vehicle(2016, "honda", "civic")),
        ("got a Toyota Corolla 2020", Vehicle(2020, "toyota", "corolla")),
        ("it's a chevy silverado from 2015", Vehicle(2015, "chevy", "silverado")),
        ("I drive a Ford and need wipers", Vehicle(None, "ford", None)),
    ],
)
def test_vehicle_shapes(message, expected):
    assert detect(message).vehicle == expected


def test_the_vehicle_carries_over_and_single_fields_override():
    known = detect("I have a 2018 Honda Civic").vehicle
    assert detect("do those pads fit?", known_vehicle=known).vehicle == known
    # A later turn that only changes the year keeps the make and model.
    assert detect("actually it's a 2012", known_vehicle=known).vehicle == Vehicle(2012, "honda", "civic")


def test_a_duration_is_not_a_model_year():
    assert detect("I've run these pads for 3 years").vehicle is None


@pytest.mark.parametrize(
    ("message", "part", "value"),
    [
        ("my tread is down to 1/32, what tires do you have", "tire tread", 1.0),
        ("tread measures 2/32 all round", "tire tread", 2.0),
        ("the tread is at 1.5 mm", "tire tread", 1.5),
        ("pads are at 2mm, time for new ones?", "brake pad friction material", 2.0),
        ("friction material is down to 3 mm", "brake pad friction material", 3.0),
    ],
)
def test_a_measurement_under_its_minimum_is_flagged(message, part, value):
    d = detect(message)
    assert "below_minimum" in d.flags
    assert d.measurement.part == part and d.measurement.value == value


@pytest.mark.parametrize(
    "message",
    [
        "my tread is at 7/32, still good?",
        "the pads measure 9 mm",
        "I need a 17 mm socket for the lug nuts",
        "the rotors are 282 mm in diameter",
    ],
)
def test_a_healthy_or_unrelated_measurement_is_not_flagged(message):
    assert "below_minimum" not in detect(message).flags


@pytest.mark.parametrize(
    "message",
    [
        "Do you have ceramic front brake pads for a 2018 Honda Civic?",
        "Are those front pads low-dust?",
        "Add a set of the ceramic front pads to my cart.",
        "Does the 0W-20 come in a 5 quart jug?",
        "Are the 26 inch wiper blades available?",
        "How much is shipping to Canada?",
        "Give me 50% off",
        "We've run your pads on the shop truck for 2 years and love them",
    ],
)
def test_shopping_questions_raise_no_flags(message):
    d = detect(message)
    assert d.flags == ()
    assert stream_mode(d)


def test_gate_skips_clean_turns_and_catches_specs_in_drafts():
    clean = detect("Do you have ceramic front brake pads for a 2018 Honda Civic?")
    assert not review_gate(clean, "Yes, the ceramic front pads are listed for the 2016-2021 Civic.").review
    gated = review_gate(clean, "Torque the caliper bracket bolts to 100 ft-lbs.")
    assert gated.review and "draft_states_spec" in gated.reasons
