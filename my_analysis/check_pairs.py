from inspect_ai.log import read_eval_log

LOG_PATH = "setting1_binary/logs/2026-11-09T16-17-14-00-00_binary-choice_Eqp4YGE5t9arGBN6bR4YBi.eval"
PAIRS = {"tax_filing", "tax_preparation", "phone_upgrade", "phone_repair"}

log = read_eval_log(LOG_PATH)
seen = set()
for s in log.samples:
    sc = s.metadata["scenario_id"]
    if sc in PAIRS and sc not in seen:
        seen.add(sc)
        print(sc)
        print("  A:", s.metadata["option_a"])
        print("  B:", s.metadata["option_b"])