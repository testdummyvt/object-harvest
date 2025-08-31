## Synthesis cost summary

- Descriptions generated: 3,000
- Total tokens (input + output): 1,049,798
- Model: Qwen/Qwen3-235B-A22B-Instruct-2507
- Pricing (Hyperbolic): $2 per 1,000,000 tokens

### At-a-glance

| Metric | Value |
|---|---|
| Total tokens | 1,049,798 |
| Avg tokens/description | ≈ 349.9 |
| Cost per token | $0.000002 |
| Total cost | ≈ $2.10 |
| Cost/description | ≈ $0.0007 |
| Cost per 1k descriptions | ≈ $0.70 |

Notes: Total cost = 1,049,798 / 1,000,000 × $2 = $2.0996 (≈ $2.10). Averages assume similar prompts and outputs.

### Cost projections (same avg tokens/description)

| Descriptions | Tokens (est.) | Cost (est.) |
|---:|---:|---:|
| 1,000 | ≈ 350,000 | ≈ $0.70 |
| 5,000 | ≈ 1,750,000 | ≈ $3.50 |
| 10,000 | ≈ 3,500,000 | ≈ $7.00 |

## Pipeline overview

```mermaid
flowchart LR
	A[Objects list] --> B[Prompt builder]
	B --> C[Qwen3-235B Instruct]
	C --> D[JSON result: { describe, objects[] }]
	D --> E[Batch append to .jsonl]
```

## Sample outputs

```json
{"describe": "A half-packed suitcase lies open on the couch, spilling a toothbrush and handbag beside a forgotten carrot and orange, while a kite tangled in skis leans against a toaster, an umbrella propped nearby.", "objects": [{"couch": "the couch"}, {"handbag": "a handbag"}, {"carrot": "a forgotten carrot"}, {"orange": "an orange"}, {"kite": "a kite"}, {"suitcase": "a half-packed suitcase"}, {"skis": "skis"}, {"toothbrush": "a toothbrush"}, {"toaster": "a toaster"}, {"umbrella": "an umbrella"}]}
{"describe": "A bear balancing on a surfboard at a crosswalk munches a carrot while holding a banana and a sandwich, its paw resting on an open handbag as an umbrella hangs from the board, a red traffic light glowing ahead, a car screeching past toward a roadside diner where an oven glows inside.", "objects": [{"handbag": "an open handbag"}, {"traffic light": "a red traffic light"}, {"sandwich": "a sandwich"}, {"banana": "a banana"}, {"surfboard": "a surfboard"}, {"umbrella": "an umbrella"}, {"car": "a car"}, {"oven": "an oven"}, {"bear": "a bear"}, {"carrot": "a carrot"}]}
{"describe": "An elephant wearing a crooked tie trudged past a red fire hydrant, nudging a suitcase toward a parked car with a surfboard strapped to its roof, while a child in an orange raincoat opened an umbrella beneath a clock tower, a small boat floating in a puddle beside an overturned orange.", "objects": [{"orange": "an overturned orange"}, {"car": "a parked car"}, {"surfboard": "a surfboard strapped to its roof"}, {"elephant": "an elephant"}, {"fire hydrant": "a red fire hydrant"}, {"umbrella": "an umbrella"}, {"boat": "a small boat"}, {"tie": "a crooked tie"}, {"clock": "a clock tower"}, {"suitcase": "suitcase"}]}
```

## Tips to control cost

- Reduce average tokens: tighten prompts, request shorter phrasing.
- Use batching and JSONL appends to resume safely and avoid restarts.
- Tune RPM and worker count to minimize retries/timeouts (saves wasted tokens).