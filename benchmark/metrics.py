import requests
from collections import defaultdict

METRICS_URL = "http://127.0.0.1:23333/metrics"

def parse_counter_total(lines, name_prefix):
    total = 0.0
    for l in lines:
        if l.startswith(name_prefix):
            try:
                total += float(l.split()[-1])
            except Exception:
                pass
    return total

def parse_histogram_components(lines, prefix):
    buckets = defaultdict(float)
    total_count = 0.0
    total_sum = 0.0

    for line in lines:
        if line.startswith(prefix + "_bucket"):
            try:
                parts = line.split()
                val = float(parts[-1])
                le_idx = line.find('le="')
                if le_idx == -1:
                    continue
                after = line[le_idx + 4:]
                le_str = after.split('"', 1)[0]
                if le_str == "+Inf":
                    continue
                le_val = float(le_str)
                buckets[le_val] += val
            except Exception:
                continue
        elif line.startswith(prefix + "_count"):
            try:
                total_count += float(line.split()[-1])
            except Exception:
                pass
        elif line.startswith(prefix + "_sum"):
            try:
                total_sum += float(line.split()[-1])
            except Exception:
                pass

    return {"sum": total_sum, "count": total_count, "buckets": dict(buckets)}

def scrape_metrics():
    r = requests.get(METRICS_URL)
    r.raise_for_status()
    return r.text.splitlines()

def take_baseline(lines):
    base = {}
    base["tokens_prompt_total"] = parse_counter_total(lines, "vllm:prompt_tokens_total")
    base["tokens_generation_total"] = parse_counter_total(lines, "vllm:generation_tokens_total")
    for prefix in [
        "vllm:time_to_first_token_seconds",
        "vllm:time_per_output_token_seconds",
        "vllm:e2e_request_latency_seconds",
        "vllm:request_prefill_time_seconds",
        "vllm:request_decode_time_seconds",
        "sglang:time_to_first_token_seconds",
        "sglang:e2e_request_latency_seconds"
    ]:
        base[prefix] = parse_histogram_components(lines, prefix)

    base["http_count"] = parse_counter_total(lines, "http_request_duration_highr_seconds_count")
    base["http_sum"] = parse_counter_total(lines, "http_request_duration_highr_seconds_sum")
    base["decode_sum"] = parse_counter_total(lines, "vllm:request_decode_time_seconds_sum")
    base["e2e_sum"] = parse_counter_total(lines, "vllm:e2e_request_latency_seconds_sum")
    return base

def histogram_delta(now, base):
    if base is None:
        return {
            "sum": now["sum"],
            "count": now["count"],
            "buckets": dict(sorted(now["buckets"].items()))
        }

    dsum = max(0.0, now["sum"] - base.get("sum", 0.0))
    dcount = max(0.0, now["count"] - base.get("count", 0.0))

    buckets = {}
    all_keys = set(now["buckets"].keys()) | set(base.get("buckets", {}).keys())
    for le in all_keys:
        nv = now["buckets"].get(le, 0.0)
        bv = base.get("buckets", {}).get(le, 0.0)
        buckets[le] = max(0.0, nv - bv)

    return {"sum": dsum, "count": dcount, "buckets": dict(sorted(buckets.items()))}


def summarize_histogram_from_delta(hdelta):
    total_count = hdelta["count"]
    total_sum = hdelta["sum"]
    buckets = hdelta["buckets"]

    if total_count <= 0:
        return None

    sorted_buckets = sorted(buckets.items())
    avg = total_sum / total_count
    median = None
    min_val = None
    max_val = None

    for le, cum in sorted_buckets:
        if min_val is None and cum > 0:
            min_val = le
        if median is None and cum >= total_count * 0.5:
            median = le
        if cum >= total_count:
            max_val = le
            break

    return {"avg": avg, "min": min_val, "median": median, "max": max_val}

def summarize_metrics_since(lines_now, base=None):
    prompt_now = parse_counter_total(lines_now, "vllm:prompt_tokens_total")
    gen_now = parse_counter_total(lines_now, "vllm:generation_tokens_total")

    if base is None:
        prompt = prompt_now
        gen = gen_now
    else:
        prompt = max(0.0, prompt_now - base.get("tokens_prompt_total", 0.0))
        gen = max(0.0, gen_now - base.get("tokens_generation_total", 0.0))

    def hist(prefix):
        now_comp = parse_histogram_components(lines_now, prefix)
        base_comp = None if base is None else base.get(prefix)
        return summarize_histogram_from_delta(histogram_delta(now_comp, base_comp))

    metrics = {}
    metrics["tokens"] = {"prompt": prompt, "generation": gen}
    metrics["ttft"] = hist("vllm:time_to_first_token_seconds") or hist("sglang:time_to_first_token_seconds")
    # metrics["tpot"] = hist("vllm:request_time_per_output_token_seconds") or hist("sglang:time_to_first_token_seconds")
    metrics["e2e"] = hist("vllm:e2e_request_latency_seconds") or hist("sglang:e2e_request_latency_seconds")
    metrics["prefill"] = hist("vllm:request_prefill_time_seconds")
    metrics["decode"] = hist("vllm:request_decode_time_seconds")

    def get_http_delta_avg():
        http_count_now = parse_counter_total(lines_now, "http_request_duration_highr_seconds_count")
        http_sum_now = parse_counter_total(lines_now, "http_request_duration_highr_seconds_sum")
        if base is None:
            dcount = http_count_now
            dsum = http_sum_now
        else:
            dcount = max(0.0, http_count_now - base.get("http_count", 0.0))
            dsum = max(0.0, http_sum_now - base.get("http_sum", 0.0))
        return (dsum / dcount) if dcount > 0 else None

    metrics["http_avg"] = get_http_delta_avg()

    decode_sum_now = parse_counter_total(lines_now, "vllm:request_decode_time_seconds_sum")
    e2e_sum_now = parse_counter_total(lines_now, "vllm:e2e_request_latency_seconds_sum")

    if base is None:
        d_decode_sum = decode_sum_now
        d_e2e_sum = e2e_sum_now
    else:
        d_decode_sum = max(0.0, decode_sum_now - base.get("decode_sum", 0.0))
        d_e2e_sum = max(0.0, e2e_sum_now - base.get("e2e_sum", 0.0))

    metrics["throughput"] = {
        "decode": (gen / d_decode_sum) if d_decode_sum > 0 else None,
        "overall": ((gen + prompt) / d_e2e_sum) if d_e2e_sum > 0 else None,
    }

    return metrics
