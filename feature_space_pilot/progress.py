"""Small progress events; console output goes to stderr, leaving JSON stdout valid."""

import sys
import time


def emit(callback, event, **fields):
    if callback:
        callback(dict(event=event, **fields))


def console_reporter():
    started = time.monotonic()

    def report(event):
        seconds = int(time.monotonic() - started)
        prefix = f"[{seconds//60:02d}:{seconds%60:02d}]"
        kind = event["event"]
        if kind == "attempt_start":
            message = f"{event['stage']}: requesting {event.get('item_id','')} {event.get('prompt_family','')} replicate={event.get('replicate','-')} attempt={event['attempt']}/{event['max_attempts']}"
        elif kind == "attempt_finished":
            message = f"{event['stage']}: {event['parse_status']} in {event['latency_seconds']:.1f}s; accounted ${event['accounted_usd']:.4f}"
        elif kind == "stage_progress":
            message = (
                f"{event['stage']}: {event['completed']}/{event['planned']} completed"
            )
        else:
            message = event.get("message", kind)
        try:
            print(prefix + " " + message, file=sys.stderr, flush=True)
        except (BrokenPipeError, OSError):
            pass

    return report
