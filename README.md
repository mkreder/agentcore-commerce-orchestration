# Agentic commerce on AgentCore: a parts counter, a commerce agent and a safety monitor

An online auto parts store that answers anything a DIYer asks. It has to get fitment right, never
invent a torque figure, always tell someone with failing brakes to stop driving, and still sell parts.
One agent that does all of that gets fragile fast. This repo splits the work:

```
            ┌────────────── low risk: skip the monitor ───────────────┐
            │                                                          ▼
 01 Detector ──▶ 02 Advisor ──▶ 03 Cart action ──▶ 04 Review gate ──▶ 05 Safety monitor ──▶ Response
 regex, flags    talks to the    commerce agent:    code decides:       one model call,      streamed
 never blocks    customer,       the only one that  second look?        no tools: pass,      over SSE
                 read-only       writes to the                          rewrite, escalate
                 tools           store, rules in code
```

It uses two Strands multi-agent patterns, each where it fits:

```
Graph (code routes the safety path):
detector ──▶ advisor ─────[review gate]──▶ monitor
         │      │
         │      └── commerce agent (agents as tools: the advisor decides when to buy)
         └─▶ classifier ──[review gate]──┘ (in parallel with the advisor; adds flags, never removes them)
```

- The turn is a [Strands Graph](https://strandsagents.com/docs/user-guide/sdk/multi-agent/graph/). The
  review gate is the condition on the edge into the monitor, so code, not a model, decides which turns
  get a second look.
- The advisor calls the commerce agent [as a tool](https://strandsagents.com/docs/user-guide/sdk/multi-agent/agents-as-tools/).
  When to add to the cart is part of the conversation, so the advisor decides. Whether adding is
  allowed is decided in code, inside the commerce agent's tools.

**Two agents act, one reviewer can't.** The advisor talks to the customer with read-only tools. The
commerce agent is the only one that writes to the store. The safety monitor is one model call with no
tools: it can change the answer, never the store. Everything that must not fail (spotting a car that
isn't safe to drive, prices, discount codes, **fitment**, the cart lock on an unsafe car) is
deterministic code that a model can't talk its way around.

The two agents are built with the [Strands harness](https://strandsagents.com/docs/user-guide/harness/).
Everything runs in one [Amazon Bedrock AgentCore Runtime](https://docs.aws.amazon.com/bedrock-agentcore/),
deployed with AWS CDK. The store is any Shopify store, over Shopify's
[UCP catalog and cart MCP](https://shopify.dev/docs/agents), or a local mock catalog.

> **Not a service manual.** The fitment table and every figure in `data/specs.json` are illustrative
> and are not verified for any real vehicle. Before you go live, have whoever signs off on your
> technical content write the service data, the detector patterns and the eval set.

## The two things code decides, and why

Most reference agents put one guardrail in code. This one puts two, because they fail differently.

| | What goes wrong | Where code says no |
|---|---|---|
| **Safety** | A customer describes failing brakes and the assistant helpfully sells them pads and overnight shipping. | `enforce()` in [`monitor.py`](src/parts_agent/monitor.py): on a critical flag, an answer that doesn't tell them to stop driving never ships. The cart tools refuse to add on the same turn. |
| **Fitment** | A customer says "just add it anyway" and the model obliges. The part arrives, doesn't fit, and the car is off the road another week. | `check_fitment()` in [`commerce/rules.py`](src/parts_agent/commerce/rules.py): the fitment table, not the conversation, decides. Called on the way *into* the cart, never on the way out. |

And one that is code because a model shouldn't be asked to do arithmetic on its own output:

| | What goes wrong | Where code says no |
|---|---|---|
| **Service figures** | The answer states 80 ft-lbs for a bolt the manual calls 25 ft-lbs. Either the part is ruined or it comes loose on the highway. | `spec_mentions()` in [`gate.py`](src/parts_agent/gate.py) finds every figure in the draft; `enforce()` compares them against the service data and replaces the answer if any figure isn't on file. |

## Where each idea lives

| Idea | Code |
|---|---|
| The orchestrator: a Strands Graph with deterministic edges | `build_turn_graph` in [`pipeline.py`](src/parts_agent/pipeline.py) |
| Detector, classifier, advisor and monitor as custom Graph nodes sharing a typed `Turn` | [`nodes.py`](src/parts_agent/nodes.py) |
| 01 Detector | Regex for brake failure, loss of control, fire and fuel, worn-out measurements, spec and diagnosis questions. Zero tokens | [`detector.py`](src/parts_agent/detector.py) |
| 01 Vehicle and wear extraction | Year/make/model pulled from the message and carried across turns; a measurement compared against a minimum in code | `Vehicle`, `_below_minimum` in [`detector.py`](src/parts_agent/detector.py) |
| 01b Classifier | A second model catches paraphrases ("the pedal feels like a sponge") and other languages. Adds flags only, in parallel with the advisor, with a timeout | [`classifier.py`](src/parts_agent/classifier.py) |
| 02 Advisor | Strands harness with **only** our tools: catalog and service data, read-only, plus the commerce agent | [`advisor.py`](src/parts_agent/advisor.py) |
| 03 Cart action | The commerce agent, called as a tool. Its cart tools refuse to add on an unsafe car, and refuse a part that doesn't fit | [`commerce/agent.py`](src/parts_agent/commerce/agent.py) |
| Store access | Allow-listed UCP tools, throttling and retries on 429 | [`commerce/store.py`](src/parts_agent/commerce/store.py) |
| Action boundary | Quantity cap, discount allow-list and the fitment refusal, in code | [`commerce/rules.py`](src/parts_agent/commerce/rules.py) |
| 04 Review gate | Review only flagged turns, plus any draft that states a service figure. Runs as the condition on the edges into the monitor | [`gate.py`](src/parts_agent/gate.py), `needs_review` in [`nodes.py`](src/parts_agent/nodes.py) |
| 05 Safety monitor | One model call, **zero** tools, structured verdict: pass, rewrite or escalate | [`monitor.py`](src/parts_agent/monitor.py) |
| Handoff | A structured payload, not the transcript | `Handoff` in [`monitor.py`](src/parts_agent/monitor.py) |
| Memory | Shared versus isolated state per agent | [`session.py`](src/parts_agent/session.py), [`specs.py`](src/parts_agent/specs.py) |
| Critical cases | Measure each layer: regex alone, then the full pipeline | [`evals/`](evals/) |
| Latency and tokens | Per-turn trace: graph path, commerce requests, reviewed?, tokens per agent, time to first token | `trace` in [`pipeline.py`](src/parts_agent/pipeline.py) |

## Quickstart: chat locally in 5 minutes

You need Python 3.10+, AWS credentials, and access to three models on Amazon Bedrock. The defaults are
Claude Sonnet 4.6 for the advisor, Claude Haiku 4.5 for the commerce agent and the monitor, and OpenAI
GPT-5.6 Luna for the classifier; set `ADVISOR_MODEL`, `COMMERCE_MODEL`, `MONITOR_MODEL` and
`CLASSIFIER_MODEL` to use others. Anthropic models need the one-time use case form in the Bedrock
console; the first call to each model subscribes the account to it in AWS Marketplace.

```bash
git clone <this repo> && cd agentic-parts-agentcore
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env                 # set AWS_PROFILE there; every setting is documented in it
set -a; source .env; set +a
python -m parts_agent.cli --demo     # replays the conversation from the talk
python -m parts_agent.cli            # or chat yourself
```

With no `SHOP_DOMAIN`, it uses the mock catalog in `data/catalog.json`. Each turn prints its trace:

```
customer> Do you have ceramic front brake pads for a 2018 Honda Civic?
assistant> Yes — the Ceramic Front Brake Pads are listed for the 2016-2021 Civic, $54.99 for the axle set. ...
  path=detector>classifier>advisor flags=[] vehicle=2018 honda civic streamed=True cart=None reviewed=False ...

customer> My brake pedal goes to the floor now and it's grinding. Can you overnight the pads?
assistant> Stop driving it. A pedal that goes to the floor, or brakes grinding metal on metal, can fail ...
  path=detector>advisor>monitor flags=['brake_failure'] streamed=False cart=None reviewed=True -> escalate escalated=True ...
```

On a turn the regex already flags as critical, the classifier is skipped, so the path is
`detector>advisor>monitor`. To watch turns in a browser instead, see
[the Streamlit page](#watch-each-turn-in-a-browser).

Tests run offline, with no model calls:

```bash
pytest
```

## Use your Shopify store

Any Shopify store exposes its catalog and cart to agents at `https://{shop}/api/ucp/mcp`. Anonymous
access works with no API key, at the lowest rate limits.

```bash
export SHOP_DOMAIN=your-store.myshopify.com
python -m parts_agent.cli
```

Then make it yours:

1. **Service data.** Put the figures for the parts you sell in `data/specs.json`, keyed by product
   handle: a `fits` table, and `bands` of figures by model year. Torque, capacity, thickness and
   interval figures are only ever quoted from this file.
2. **Fitment.** The `fits` rows are what `check_fitment()` enforces. A part with an empty `fits` list
   is treated as universal; a part with no entry at all has no service data and the advisor is told
   not to state any figure for it.
3. **Agent profile.** Every UCP request carries your agent's profile URL. The default is Shopify's
   hosted test profile (`valid-with-capabilities.json`). Before production,
   [host your own profile](https://shopify.dev/docs/agents/profiles) and set `UCP_AGENT_PROFILE`.
4. **Discount codes.** Set `DISCOUNT_CODES=WELCOME10:10,VIP15:15`. No other discount exists.
5. **Detector patterns.** Add your domain's red flags to `detector.py`, and add each one to
   `evals/cases.jsonl`.

Checkout ends on the store's own cart page (`continue_url`). Agent-completed checkout through UCP
`create_checkout` is phase 2.

## Deploy to AgentCore Runtime with AWS CDK

The whole turn runs in one AgentCore Runtime. The classifier, the commerce agent and the monitor are
in-process calls: each one costs a model call, never an extra network hop between services.

**No Docker needed.** By default the stack uses AgentCore **direct code deployment**. During
`cdk deploy`, a local bundler runs `uv pip install --python-platform aarch64-manylinux_2_28` to fetch
Linux ARM64 wheels, adds the package, the service data and a `main.py` entrypoint, and CDK uploads the
zip. You need [uv](https://docs.astral.sh/uv/getting-started/installation/), Node.js 20+ and
Python 3.11+.

```bash
pip install -e ".[cdk]"               # aws-cdk-lib, in the same venv
cd infra
npm install                           # pins the CDK CLI version this aws-cdk-lib needs
npx cdk bootstrap --profile <profile>                     # once per account and region
SHOP_DOMAIN=your-store.myshopify.com npx cdk deploy --profile <profile>
```

Leave `SHOP_DOMAIN` out to deploy with the mock catalog. The runtime gets these settings from your
shell at deploy time, the same way: `UCP_AGENT_PROFILE`, `ADVISOR_MODEL`, `COMMERCE_MODEL`,
`MONITOR_MODEL`, `CLASSIFIER`, `CLASSIFIER_MODEL`, `CLASSIFIER_TIMEOUT_S`,
`CLASSIFIER_REASONING_EFFORT`, `DISCOUNT_CODES`, `MAX_QTY_PER_LINE` and `ENFORCE_FITMENT`. Nothing
else from your shell or your `.env` reaches it.

To remove everything, run `npx cdk destroy --profile <profile>`. AgentCore creates the runtime's log
group outside CloudFormation, so delete `/aws/bedrock-agentcore/runtimes/parts_agent-*` in CloudWatch
Logs by hand.

The stack outputs `RuntimeArn`. Chat with it:

```bash
AWS_PROFILE=<profile> python scripts/invoke.py <RuntimeArn> "Do you have front brake pads for a 2018 Honda Civic?"
AWS_PROFILE=<profile> python scripts/invoke.py <RuntimeArn> "Add a set to my cart" --session <session id printed above>
```

### Watch each turn in a browser

A Streamlit page chats with the deployed runtime, or with the orchestrator on your machine, and shows
what happened in every turn: the graph with the nodes that ran, the regex and classifier flags, the
vehicle on file, the fitment check, the review gate, the monitor's verdict and the draft it repaired,
the commerce agent's requests, tokens per agent and latency.

```bash
pip install -e ".[ui]"
RUNTIME_ARN=<RuntimeArn> AWS_PROFILE=<profile> streamlit run ui/streamlit_app.py
```

Leave `RUNTIME_ARN` out and pick **Local** in the sidebar to run without a deploy. The sidebar has the
demo conversation as buttons.

### Container and observability

To deploy as a container instead, run `npx cdk deploy -c artifact=container`. That builds the root
`Dockerfile` as an ARM64 image and needs Docker or Finch.

Responses stream as Server-Sent Events:

| Event | Meaning |
|---|---|
| `{"type": "text", "delta": "..."}` | Answer text, as it is generated |
| `{"type": "replace", "text": "..."}` | A streamed answer the monitor repaired after the fact |
| `{"type": "done", "trace": {...}}` | Graph path, regex and classifier flags, vehicle, review, verdict, commerce requests, tokens per agent, latency |
| `{"type": "error", "message": "..."}` | The request had no `prompt` |

**Observability.** The runtime starts under ADOT (`opentelemetry-instrument`). To get traces in the
AgentCore Observability dashboards,
[enable CloudWatch Transaction Search](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability-configure.html)
once in your account, then deploy with `npx cdk deploy -c tracing=true`. Without Transaction Search, a
deploy with tracing fails at `TracesDelivery`, so tracing is off by default.

## Evals

```bash
python evals/run_evals.py                # regex layer only: offline, zero tokens
python evals/run_evals.py --classifier   # regex + classifier, one model call per case
python evals/run_evals.py --full         # whole pipeline against Bedrock
```

The critical set has zero tolerance: every case must end in "stop driving it". The shopping set
measures over-refusal just as seriously. A safe assistant that refuses "are these pads low-dust?" is
still a failure.

The set has 20 critical cases and 12 shopping questions, a few of each in Spanish. The regex layer
misses some cases on purpose: paraphrases like "feels like stepping on a sponge" or "something under
the front end let go", and every Spanish one. The classifier is there to catch them, and measuring each
layer shows whether it does. When the full pipeline misses a case, you add a pattern to the detector or
tune a prompt, and the case stays in the set for good.

### Detection layers

What each detection layer flags on its own, before any agent runs: 20 critical cases (16 in English, 4
in Spanish) and 12 shopping questions (10 in English, 2 in Spanish). A critical case passes when it gets
a critical flag; a shopping question fails when it gets any flag, because that sends it to review.

The regex column is measured and reproducible offline (`python evals/run_evals.py`, zero tokens). The
model columns depend on your account, your region and your model choice, so they are **left blank on
purpose**: run `--classifier` and fill in what you measure. Don't ship numbers you didn't take.

| | Regex only | Regex + your classifier |
|---|---|---|
| Critical cases flagged as critical | 12/20 | |
| Paraphrases (4) and Spanish (4) caught | 0/8 | |
| Shopping questions sent to review | 0/12 | |
| Latency per message, p50 | ~0 ms | |
| Tokens per message | 0 | |

The eight the regex misses are the point of the second layer: four English paraphrases that describe a
failure without using any of its words, and four Spanish messages. `--classifier` measures the
configured classifier; set `CLASSIFIER_MODEL` to compare two.

### What to measure before you put a number on a slide

`--full` prints the four numbers worth reporting, per run:

- **Critical escalated** and **shopping answered without refusal.** Both, always. One without the other
  is not a result.
- **Turns reviewed by the monitor.** The share of turns that pay for a second model call. Every review
  should trace back to a flag you can name.
- **Tokens per turn (median)** and **time to first token p50.** Report streamed and buffered turns
  separately: a flagged turn is buffered on purpose, so its first token waits for the monitor.

> **Check your quotas before you measure.** Accounts can start with 10 requests per minute per model,
> and one turn makes several model calls. Above that rate Bedrock returns `ThrottlingException` and the
> SDK retries with backoff, so wall-clock latency stops reflecting the pipeline and starts reflecting
> the queue. Raise the requests-per-minute quotas in Service Quotas before you put latency numbers on a
> slide or run a live demo.

## Design notes

- **Two detection layers, one rule.** The regex is the floor: instant, deterministic, and nothing a
  customer types can argue with it. The classifier catches what the regex can't read: paraphrases,
  other languages and typos. It can only add flags. A wrong answer from it costs an extra review, never
  a missed escalation. It runs in parallel with the advisor, and only when it can change something: if
  the regex already found a critical flag, the turn is reviewed anyway and the classifier is skipped. It
  is cut off after `CLASSIFIER_TIMEOUT_S`. If it times out or fails, the turn goes to the monitor: not
  knowing means review, never "looks fine". The cart lock then keeps the regex's flags, so a slow
  Bedrock call doesn't block a purchase. If it flags a turn whose answer is already streaming, the
  monitor repairs it with a `replace` event.
- **Which multi-agent pattern where.** Strands has four, and the choice is about who should decide. The
  safety path is a Graph: a car with no brakes must be escalated every time, not most of the time, so
  the edges are Python. Swarm, Workflow or calling the monitor as a tool would all let a model pick that
  path. Buying is different: when to add to the cart is conversation, so the advisor calls the commerce
  agent as a tool and gets the result in the same turn ("those pads aren't listed for a 2012, want the
  ones that are?"). Whether adding is allowed stays in code, in the commerce agent's tools.
- **The fitment gate is on the write path, not in the prompt.** The advisor is told to ask for the
  vehicle and to respect fitment, and it mostly will. `check_fitment()` is what makes "just add it
  anyway" fail: it runs inside `update_cart`, before the store is touched, and only on the way up.
  Taking items out never has to fit. It blocks only an explicit `does_not_fit`; "no vehicle on file" and
  "universal" both go through, and the tool tells the advisor which it was so it can ask. Set
  `ENFORCE_FITMENT=false` to demo what the prompt alone does.
- **The monitor is a call, not an agent.** It has no tools, no memory and no loop. Code looks up what it
  needs (flags, the vehicle, the service figures that vehicle is entitled to, the wear measurement, the
  cart action) and hands it over, so the safety check never depends on a model deciding what to look up.
  It still uses a plain Strands `Agent` with `tools=[]`: that's Strands' structured call, with Pydantic
  validation, token metrics and spans inside the graph.
- **Custom nodes and a typed handoff.** By default a Graph node receives the original task plus the text
  output of every upstream node. That is a transcript, and the monitor should see a structured payload.
  So each node is a small `MultiAgentBase` that reads and writes a typed `Turn` passed in
  `invocation_state`. Wrapping the advisor also keeps its conversation across turns: one advisor and one
  graph per session.
- **A worn-out part still gets sold a part.** This is where an auto parts store differs from most
  safety-reviewed assistants. Tread at 1/32 in gets a warning that `enforce()` prepends whether or not
  the model wrote one, and then the answer goes right on to recommend tires, because that *is* the fix.
  Only the critical set (brakes, control, fire) suppresses the sale.
- **Product sizes are not service figures.** `spec_mentions()` subtracts product and tool sizes before
  it looks for a figure, so "a 5 quart jug", "a 17 mm socket", "26 inch blades" and "rated to 60,000
  miles" don't read as specs, while "takes 3.7 quarts", "80 ft-lbs" and "every 7,500 miles" do. This is
  the regression that over-refusal lives in: get it wrong and the store can't describe its own products.
  There are tests for every one of those strings.
- **Why the harness, narrowed.** `create_harness()` gives you a capable agent with a shell, files, web,
  memory and a coding-agent system prompt. A parts counter needs none of that. The advisor and the
  commerce agent pass `builtin_tools=[]`, `builtin_plugins=[]`, `memory=False` and their own
  `system_prompt`. The harness still handles model resolution, prompt caching and sessions. The test
  suite asserts exactly which tools each agent sees, and that only the commerce agent can write to the
  store.
- **Streaming versus review.** If the detector flags a turn, the turn is buffered, because the customer
  must never see a draft the monitor is about to rewrite. Every other turn streams immediately. If the
  gate later finds a service figure in a streamed draft, the monitor's repair goes out as a `replace`
  event.
- **Fail closed.** If the monitor errors or gets it wrong on a critical turn, `enforce()` in
  `monitor.py` replaces the answer with a fixed escalation. A torque figure that isn't on file never
  ships.
- **The spec check reads English.** `spec_mentions()` in `gate.py` finds figures like "80 ft-lbs" or
  "3.7 quarts" and tells product sizes apart, in English. A figure written in Spanish ("80 libras-pie")
  isn't detected, so it isn't checked against the service data unless the turn was already flagged. For
  a multilingual store, the robust fix is to let the monitor list the figures it reads, as structured
  output, and keep the comparison against the service data in code: the model reads, code decides.
- **The vehicle parser is a regex, with the limits of one.** It reads a year, a known make and the token
  after it, and carries each field across turns. It does not read a VIN, a trim or an engine code, and a
  model-only message ("what about the Corolla") won't update the model. A wrong vehicle shows up in the
  trace and in `<turn_context>`, and the worst it does is refuse a part that would have fitted, which is
  the safe direction. For production, take the vehicle from the customer's garage on the storefront
  rather than from the chat, and use the chat only to confirm it.
- **No adding on an unsafe car.** The monitor runs after the advisor, so the cart lock can't wait for
  its verdict. On a turn the detector flags as critical, the commerce agent's tools refuse to add. After
  the monitor escalates, they refuse for the rest of the session. Removing items and checking out still
  work.
- **The commerce agent starts each request fresh.** The cart lives in code (`CommerceState`), not in a
  transcript, so every request from the advisor starts from an empty conversation. Requests run one at a
  time, even if the advisor calls tools in parallel.
- **State.** Session state is in process, which is enough because AgentCore gives each session its own
  microVM. For durable state, map the four blocks in `session.py` to AgentCore Memory namespaces:
  `/sessions/{id}/shared`, `/advisor`, `/commerce` and `/monitor`.
- **Bedrock Guardrails.** It is complementary, not a replacement. Content filters make a cheap baseline
  in front of everything, and Automated Reasoning checks fit claims like "listed for 2016-2021".
  Guardrails can't catch an omission: a missing "stop driving it" breaks no rule.

## License

MIT-0
