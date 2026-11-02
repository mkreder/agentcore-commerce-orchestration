"""The orchestrator end to end, with scripted agents: no model calls, no AWS."""

import asyncio
import dataclasses
from types import SimpleNamespace

import pytest

from parts_agent.advisor import build_advisor
from parts_agent.commerce.agent import as_tool, build_commerce_agent
from parts_agent.commerce.store import MockStore
from parts_agent.config import Settings
from parts_agent.detector import Vehicle
from parts_agent.monitor import Verdict, enforce
from parts_agent.pipeline import Orchestrator

SETTINGS = Settings(
    store_mode="mock",
    shop_domain=None,
    ucp_agent_profile="",
    advisor_model=None,
    commerce_model=None,
    monitor_model=None,
    classifier="off",
    classifier_model=None,
    classifier_timeout_s=3.0,
    classifier_reasoning_effort="none",
    discount_codes={"WELCOME10": 10},
    max_qty_per_line=6,
    enforce_fitment=True,
    session_dir="/tmp/parts-agent-test-sessions",
)

PADS, ROTORS, OIL, WIPERS = (
    "gid://mock/ProductVariant/101",
    "gid://mock/ProductVariant/201",
    "gid://mock/ProductVariant/401",
    "gid://mock/ProductVariant/601",
)


class ScriptedAdvisor:
    def __init__(self, replies):
        self.replies = list(replies)
        self.messages = []

    async def stream_async(self, prompt):
        reply = self.replies.pop(0)
        self.messages += [{"role": "user", "content": [{"text": prompt}]}, {"role": "assistant", "content": [{"text": reply}]}]
        for word in reply.split(" "):
            yield {"data": word + " "}
        yield {"result": SimpleNamespace(metrics=SimpleNamespace(accumulated_usage={"totalTokens": 100}))}


class ScriptedMonitor:
    def __init__(self, verdict):
        self.verdict = verdict
        self.handoffs = []

    async def review(self, handoff):
        self.handoffs.append(handoff)
        return enforce(handoff, self.verdict), {"totalTokens": 50}


class ScriptedClassifier:
    name = "scripted"

    def __init__(self, flags, delay=0.0, fail=False):
        self.flags, self.delay, self.fail, self.calls = flags, delay, fail, 0

    async def classify(self, message, vehicle):
        self.calls += 1
        await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("ThrottlingException")
        return list(self.flags), {"totalTokens": 30}


def _text(tool_result):
    return tool_result["content"][0]["text"]


async def run(orch, session, text):
    events = [e async for e in orch.run_turn(session, text)]
    return events, events[-1]["trace"]


async def test_low_risk_turn_streams_and_skips_the_monitor():
    advisor = ScriptedAdvisor(["Yes, the ceramic front pads are listed for the 2016-2021 Civic."])
    monitor = ScriptedMonitor(Verdict(action="pass", answer="unused"))
    orch = Orchestrator(SETTINGS, store=MockStore(), advisor_factory=lambda *_: advisor, monitor=monitor)

    events, trace = await run(orch, "s1", "Do you have ceramic front brake pads for a 2018 Honda Civic?")

    assert trace["streamed"] and not trace["review"]["reviewed"]
    assert trace["path"] == ["detector", "advisor"]  # the review-gate edge stayed closed
    assert trace["vehicle"] == "2018 honda civic"
    assert monitor.handoffs == []
    assert sum(e["type"] == "text" for e in events) > 1  # streamed in chunks


async def test_brake_failure_is_buffered_and_escalated_even_if_the_monitor_misses_it():
    draft = "Our ceramic pads are a great upgrade and they ship overnight. Want me to add a set?"
    advisor = ScriptedAdvisor([draft, "Sure, adding them now."])
    monitor = ScriptedMonitor(Verdict(action="pass", answer=draft))  # a monitor that got it wrong
    orch = Orchestrator(SETTINGS, store=MockStore(), advisor_factory=lambda *_: advisor, monitor=monitor)

    events, trace = await run(orch, "s2", "My brake pedal goes to the floor and it's grinding. Can you overnight the pads?")

    texts = [e for e in events if e["type"] == "text"]
    assert len(texts) == 1 and "Stop driving it" in texts[0]["delta"]  # the draft never reached the customer
    assert trace["review"]["verdict"] == "escalate" and trace["escalated"]
    assert trace["path"] == ["detector", "advisor", "monitor"]
    assert monitor.handoffs[0].detector_flags == ["brake_failure"]
    assert advisor.messages[-1]["content"][0]["text"] == texts[0]["delta"]  # advisor history holds what was sent

    _, trace2 = await run(orch, "s2", "ok, add the pads to my cart anyway")
    assert trace2["escalated"]  # sticky: the cart tools now refuse to add (see the tools test)


async def test_streamed_turn_with_a_service_figure_is_repaired_with_replace():
    advisor = ScriptedAdvisor(["Great pads. Torque the bracket bolts to 137 ft-lbs and you're done."])
    repaired = "Great pads. Use the torque figures in the service manual for your car."
    monitor = ScriptedMonitor(Verdict(action="rewrite", answer=repaired))
    orch = Orchestrator(SETTINGS, store=MockStore(), advisor_factory=lambda *_: advisor, monitor=monitor)

    events, trace = await run(orch, "s3", "Tell me about your ceramic front pads")

    assert trace["streamed"] and "draft_states_spec" in trace["review"]["reasons"]
    assert events[-2] == {"type": "replace", "text": repaired}


async def test_a_worn_measurement_reaches_the_monitor_as_structured_data():
    draft = "We have a few all-season options in your size."
    advisor = ScriptedAdvisor([draft])
    monitor = ScriptedMonitor(Verdict(action="pass", answer=draft))
    orch = Orchestrator(SETTINGS, store=MockStore(), advisor_factory=lambda *_: advisor, monitor=monitor)

    events, trace = await run(orch, "s5", "my tread is down to 1/32, what tires do you have?")

    assert trace["flags"] == ["below_minimum"] and not trace["streamed"]
    assert monitor.handoffs[0].measurement["value"] == 1.0
    assert monitor.handoffs[0].measurement["minimum"] == 2.0
    # The warning is code: it is there whether or not the model wrote one, and the sales answer survives.
    assert trace["review"]["verdict"] == "rewrite"
    assert "enforced_wear_warning" in trace["review"]["verdict_reasons"]
    sent = [e for e in events if e["type"] == "text"][0]["delta"]
    assert "worn out" in sent and "all-season options" in sent
    assert trace["measurement"] == "tire tread at 1/32 in (minimum 2/32 in)"


@pytest.fixture
def real_commerce_tools(monkeypatch):
    """The real commerce agent's tools, called directly (no model): the action boundary is code."""
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    orch = Orchestrator(SETTINGS, store=MockStore(), monitor=ScriptedMonitor(None))
    session = orch.sessions.get("tools")
    session.shared.vehicle = Vehicle(2018, "honda", "civic")
    agent = build_commerce_agent(SETTINGS, orch.store, session)
    return agent, session


def test_only_the_commerce_agent_can_write_to_the_store(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    store, session = MockStore(), Orchestrator(SETTINGS, store=MockStore(), monitor=ScriptedMonitor(None)).sessions.get("t")
    advisor = build_advisor(SETTINGS, store, session)
    commerce = build_commerce_agent(SETTINGS, store, session)
    assert sorted(advisor.tool_names) == ["commerce_agent", "get_product", "read_specs", "search_products"]
    assert sorted(commerce.tool_names) == ["apply_discount_code", "checkout_link", "get_product", "update_cart", "view_cart"]


async def test_a_critical_turn_cannot_add_to_the_cart_before_the_monitor_runs(real_commerce_tools):
    """The monitor runs after the advisor, so the guard can't wait for its escalation."""
    agent, session = real_commerce_tools
    session.shared.turn_flags = ("brake_failure",)
    blocked = agent.tool.update_cart(variant_id=PADS, quantity=1, record_direct_tool_call=False)
    assert "isn't safe to drive" in _text(blocked) and not session.commerce.lines

    session.shared.turn_flags = ("spec_question",)  # asking for a torque figure is not an emergency
    ok = agent.tool.update_cart(variant_id=PADS, quantity=1, record_direct_tool_call=False)
    assert "54.99 USD" in _text(ok)


async def test_the_cart_refuses_a_part_that_does_not_fit_the_vehicle(real_commerce_tools):
    agent, session = real_commerce_tools
    ok = agent.tool.update_cart(variant_id=PADS, quantity=1, record_direct_tool_call=False)
    assert ok["status"] == "success" and "54.99 USD" in _text(ok)

    session.shared.vehicle = Vehicle(2012, "honda", "civic")
    misfit = agent.tool.update_cart(variant_id=ROTORS, quantity=1, record_direct_tool_call=False)
    assert "doesn't fit a 2012 honda civic" in _text(misfit)
    assert ROTORS not in session.commerce.lines

    # Universal-fit parts go through whatever the vehicle is, and say so.
    universal = agent.tool.update_cart(variant_id=OIL, quantity=1, record_direct_tool_call=False)
    assert "universal fit" in _text(universal)

    # Taking something out never has to fit.
    removed = agent.tool.update_cart(variant_id=PADS, quantity=0, record_direct_tool_call=False)
    assert removed["status"] == "success" and PADS not in session.commerce.lines


async def test_with_no_vehicle_on_file_the_tool_adds_and_says_fitment_is_unchecked(real_commerce_tools):
    agent, session = real_commerce_tools
    session.shared.vehicle = None
    added = agent.tool.update_cart(variant_id=PADS, quantity=1, record_direct_tool_call=False)
    assert added["status"] == "success"
    assert "no vehicle on file" in _text(added)


async def test_commerce_tools_enforce_business_rules(real_commerce_tools):
    agent, session = real_commerce_tools

    ok = agent.tool.update_cart(variant_id=PADS, quantity=2, record_direct_tool_call=False)
    assert ok["status"] == "success" and "109.98 USD" in _text(ok)

    too_many = agent.tool.update_cart(variant_id=PADS, quantity=50, record_direct_tool_call=False)
    assert "limits this item" in _text(too_many)

    fifty_off = agent.tool.apply_discount_code(code="50OFF", record_direct_tool_call=False)
    assert "isn't a valid discount code" in _text(fifty_off)

    out_of_stock = agent.tool.update_cart(variant_id=WIPERS, quantity=1, record_direct_tool_call=False)
    assert "out of stock" in _text(out_of_stock)

    # After an escalation the advisor can't add, but the customer can still remove items and check out.
    session.monitor.escalated = True
    blocked = agent.tool.update_cart(variant_id=ROTORS, quantity=1, record_direct_tool_call=False)
    assert "escalated" in _text(blocked)
    fewer = agent.tool.update_cart(variant_id=PADS, quantity=1, record_direct_tool_call=False)
    assert "54.99 USD" in _text(fewer)
    assert list(session.commerce.lines) == [PADS]
    assert "/cart/c/" in _text(agent.tool.checkout_link(record_direct_tool_call=False))


async def test_a_passing_verdict_ships_the_draft_untouched():
    draft = "The caliper bracket bolts are 100 ft-lbs on a 2018 Civic.\n\nWant me to add the pads?"
    advisor = ScriptedAdvisor([draft])
    # Models sometimes re-encode the text they pass through; a pass must not trigger a replace.
    monitor = ScriptedMonitor(Verdict(action="pass", answer=draft.replace("\n", "\\n")))
    orch = Orchestrator(SETTINGS, store=MockStore(), advisor_factory=lambda *_: advisor, monitor=monitor)

    events, trace = await run(orch, "s4", "What torque do the caliper bracket bolts take on my 2018 Honda Civic?")

    assert trace["review"]["verdict"] == "pass"
    assert not any(e["type"] == "replace" for e in events)


async def test_commerce_agent_as_a_tool_starts_each_request_fresh_and_counts_tokens():
    class FakeCommerce:
        def __init__(self):
            self.messages, self.seen = [], []

        async def invoke_async(self, request):
            self.seen.append(list(self.messages))
            self.messages += [{"role": "user", "content": [{"text": request}]}]
            usage = {"inputTokens": 80, "outputTokens": 20, "totalTokens": 100}
            return _Result("Added. Total 54.99 USD.", usage)

    fake, session = FakeCommerce(), Orchestrator(SETTINGS, store=MockStore(), monitor=ScriptedMonitor(None)).sessions.get("t")
    ask = as_tool(fake, session)
    session.commerce.start_turn()
    assert await ask(request=f"add 1 of {PADS}") == "Added. Total 54.99 USD."
    await ask(request="show the cart")
    assert fake.seen == [[], []]  # no history carried between requests
    assert session.commerce.turn_requests == [f"add 1 of {PADS}", "show the cart"]
    assert session.commerce.turn_usage["totalTokens"] == 200


class _Result:
    def __init__(self, text, usage):
        self.text, self.metrics = text, SimpleNamespace(accumulated_usage=usage)

    def __str__(self):
        return self.text


async def test_the_classifier_catches_a_paraphrase_the_regex_misses():
    draft = "Our ceramic pads are a great upgrade, and they're in stock."
    advisor = ScriptedAdvisor([draft])
    monitor = ScriptedMonitor(Verdict(action="pass", answer=draft))  # even if the monitor misses it
    classifier = ScriptedClassifier(["brake_failure"])
    orch = Orchestrator(SETTINGS, store=MockStore(), advisor_factory=lambda *_: advisor, monitor=monitor, classifier=classifier)

    events, trace = await run(orch, "c1", "the brakes feel like stepping on a sponge and then nothing happens")

    assert trace["flags"] == []  # the regex alone found nothing
    assert trace["classifier"]["added"] == ["brake_failure"]
    assert set(trace["path"]) == {"detector", "advisor", "classifier", "monitor"}
    assert monitor.handoffs[0].detector_flags == ["brake_failure"]
    assert trace["review"]["verdict"] == "escalate" and trace["escalated"]
    assert "Stop driving it" in events[-2].get("text", "")


async def test_the_classifier_can_add_flags_but_never_remove_them():
    draft = "On a 2018 Civic the caliper bracket bolts are 100 ft-lbs."
    advisor = ScriptedAdvisor([draft])
    monitor = ScriptedMonitor(Verdict(action="pass", answer=draft))
    orch = Orchestrator(SETTINGS, store=MockStore(), advisor_factory=lambda *_: advisor, monitor=monitor, classifier=ScriptedClassifier([]))

    _, trace = await run(orch, "c2", "What torque do the caliper bracket bolts take on my 2018 Honda Civic?")

    assert trace["flags"] == ["spec_question"] and trace["classifier"]["flags"] == []
    assert trace["review"]["reviewed"]  # the regex's flag stands


async def test_the_classifier_is_skipped_when_the_regex_already_found_a_critical_flag():
    advisor = ScriptedAdvisor(["Our ceramic pads are in stock."])
    monitor = ScriptedMonitor(Verdict(action="pass", answer="unused"))
    classifier = ScriptedClassifier([])
    orch = Orchestrator(SETTINGS, store=MockStore(), advisor_factory=lambda *_: advisor, monitor=monitor, classifier=classifier)

    _, trace = await run(orch, "c4", "My brake pedal goes to the floor and it's grinding. Which pads ship fastest?")

    assert classifier.calls == 0 and trace["classifier"] is None
    assert trace["path"] == ["detector", "advisor", "monitor"] and trace["escalated"]


@pytest.mark.parametrize("classifier, reason", [
    (ScriptedClassifier(["loss_of_control"], delay=0.5), "classifier_timeout"),
    (ScriptedClassifier([], fail=True), "classifier_error"),
])
async def test_a_classifier_that_times_out_or_fails_sends_the_turn_to_review(classifier, reason):
    advisor = ScriptedAdvisor(["Yes, they're low-dust."])
    monitor = ScriptedMonitor(Verdict(action="pass", answer="Yes, they're low-dust."))
    settings = dataclasses.replace(SETTINGS, classifier_timeout_s=0.1)
    orch = Orchestrator(settings, store=MockStore(), advisor_factory=lambda *_: advisor, monitor=monitor, classifier=classifier)

    events, trace = await run(orch, "c5", "Are those front pads low-dust?")

    assert trace["review"]["reviewed"] and reason in trace["review"]["reasons"]
    assert len(monitor.handoffs) == 1 and not any(e["type"] == "replace" for e in events)
    assert trace["classifier"]["latency_ms"] < 400


async def test_a_quiet_classifier_keeps_shopping_turns_off_the_monitor():
    advisor = ScriptedAdvisor(["Yes, they're low-dust."])
    monitor = ScriptedMonitor(None)
    orch = Orchestrator(SETTINGS, store=MockStore(), advisor_factory=lambda *_: advisor, monitor=monitor, classifier=ScriptedClassifier([]))

    _, trace = await run(orch, "c3", "Are those front pads low-dust?")

    assert not trace["review"]["reviewed"] and monitor.handoffs == []
    assert set(trace["path"]) == {"detector", "advisor", "classifier"}


async def test_the_cart_lock_waits_for_the_classifier(real_commerce_tools):
    agent, session = real_commerce_tools
    update_cart = agent.tool_registry.registry["update_cart"]
    session.shared.turn_flags = ()
    session.shared.flags_settled.clear()

    async def classifier_finishes_late():
        await asyncio.sleep(0.05)
        session.shared.turn_flags = ("fire_or_fuel",)
        session.shared.flags_settled.set()

    asyncio.get_running_loop().create_task(classifier_finishes_late())
    blocked = await update_cart(variant_id=PADS, quantity=1)
    assert "isn't safe to drive" in blocked["error"] and not session.commerce.lines


def test_reasoning_effort_is_only_sent_to_openai_models():
    from parts_agent.classifier import _request_fields

    assert _request_fields("global.openai.gpt-5.6-luna", "none") == {"additional_request_fields": {"reasoning": {"effort": "none"}}}
    assert _request_fields("global.anthropic.claude-haiku-4-5-20251001-v1:0", "none") == {}
    assert _request_fields("global.openai.gpt-5.6-luna", None) == {}
