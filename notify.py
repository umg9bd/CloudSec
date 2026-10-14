"""
notify.py -- emails every alert at or above the alert threshold to the admin, through Gmail.

Settings (detection runs the same with or without them):
  CLOUDSEC_SMTP_USER       the Gmail address the alerts are sent from
  CLOUDSEC_SMTP_PASSWORD   that account's Gmail app password (Google Account -> Security ->
                           2-Step Verification -> App passwords; not the normal password)
  admin address            typed in on the dashboard (saved to notify_settings.json), or
                           CLOUDSEC_ADMIN_EMAIL when nothing was saved there
Optional: CLOUDSEC_SMTP_HOST (smtp.gmail.com), CLOUDSEC_SMTP_PORT (465, SSL),
          CLOUDSEC_EMAIL_MAX_PER_HOUR (60; Gmail locks accounts that send in bulk).
run.cmd / run.sh hand the variables in email.env (git-ignored) to the container.

Never in the way of detection: with a setting missing an alert is only not emailed (said once);
a failed send is logged and dropped. Emails go out on one background thread, so a slow or
unreachable mail server never delays scoring. The admin address is re-read for every email, so a
change on the dashboard applies to the next alert without a restart.
"""
from __future__ import annotations

import json
import os
import queue
import re
import smtplib
import ssl
import threading
import time
from email.message import EmailMessage
from typing import Optional

ROOT = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH = os.path.join(ROOT, "notify_settings.json")
DEFAULT_HOST, DEFAULT_PORT, DEFAULT_MAX_PER_HOUR = "smtp.gmail.com", 465, 60
TOP_EVENTS_IN_EMAIL = 10
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def valid_email(address: str) -> bool:
    return bool(address) and bool(_EMAIL_RE.match(address.strip()))


def load_admin_email(path: str = SETTINGS_PATH) -> Optional[str]:
    """The dashboard's saved address, else CLOUDSEC_ADMIN_EMAIL, else None."""
    try:
        with open(path, encoding="utf-8") as f:
            saved = (json.load(f).get("admin_email") or "").strip()
        if saved:
            return saved
    except (OSError, ValueError):
        pass
    env = os.environ.get("CLOUDSEC_ADMIN_EMAIL", "").strip()
    return env or None


def save_admin_email(address: str, path: str = SETTINGS_PATH) -> None:
    """Saves the admin address ("" turns emailing off). Raises ValueError on a malformed one."""
    address = (address or "").strip()
    if address and not valid_email(address):
        raise ValueError(f"not an email address: {address!r}")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"admin_email": address}, f, indent=2)
    os.replace(tmp, path)   # never a half-written file for the pipeline to read


def sender() -> Optional[dict]:
    user = os.environ.get("CLOUDSEC_SMTP_USER", "").strip()
    password = os.environ.get("CLOUDSEC_SMTP_PASSWORD", "").replace(" ", "")  # Google shows it in 4 groups
    if not user or not password:
        return None
    return {"user": user, "password": password,
            "host": os.environ.get("CLOUDSEC_SMTP_HOST", DEFAULT_HOST),
            "port": int(os.environ.get("CLOUDSEC_SMTP_PORT", DEFAULT_PORT))}


def status(path: str = SETTINGS_PATH) -> dict:
    s, admin = sender(), load_admin_email(path)
    return {"sender": s["user"] if s else None, "admin_email": admin, "ready": bool(s and admin)}


def format_alert(alert: dict) -> EmailMessage:
    """Subject + plain-text body for one pipeline alert (pipeline.Pipeline.emit's dict)."""
    events = alert.get("events") or []
    top = events[0]["event_name"] if events else "?"
    msg = EmailMessage()
    msg["Subject"] = (f"[CloudSec] Risk {alert['max_risk_score']:.2f}/10: {alert['principal']} "
                      f"({top})")
    lines = [
        "CloudSec raised an alert.",
        "",
        f"Principal:      {alert['principal']}",
        f"ARN:            {alert.get('principal_arn') or 'unknown'}",
        f"When:           {alert['first_seen']} to {alert['last_seen']}",
        f"Highest risk:   {alert['max_risk_score']:.2f}/10 (alerts at {alert['threshold'] * 10:.2f}/10)",
        f"Flagged events: {alert['n_flagged_events']}",
        f"Source file:    {alert['source_file']}",
    ]
    if alert.get("fast_lane_reasons"):
        lines.append(f"Rule alerts:    {'; '.join(alert['fast_lane_reasons'])}")
    explanations = alert.get("explanations") or []
    if explanations and explanations[0].get("summary"):
        lines += ["", "Why it was flagged:", f"  {explanations[0]['summary']}"]
    if events:
        lines += ["", f"Riskiest events (up to {TOP_EVENTS_IN_EMAIL}):"]
        for e in events[:TOP_EVENTS_IN_EMAIL]:
            lines.append(f"  {str(e['timestamp'])[:19]}  {e['event_name']:<32} risk {e['risk_score']:5.2f}/10"
                         f"  target {e.get('target') or '-'}")
    lines += ["", f"Alert id: {alert['alert_id']}. The full alert, with both models' explanations, is on "
                  f"the CloudSec dashboard."]
    msg.set_content("\n".join(lines))
    return msg


def _send(messages: list, to: str, s: dict) -> None:
    """Sends messages over one SMTP connection. Raises on any failure."""
    ctx = ssl.create_default_context()
    if s["port"] == 465:
        server = smtplib.SMTP_SSL(s["host"], s["port"], timeout=20, context=ctx)
    else:
        server = smtplib.SMTP(s["host"], s["port"], timeout=20)
        server.starttls(context=ctx)
    with server:
        server.login(s["user"], s["password"])
        for m in messages:
            del m["From"], m["To"]
            m["From"], m["To"] = s["user"], to
            server.send_message(m)


def send_test(path: str = SETTINGS_PATH) -> tuple[bool, str]:
    """Sends one test email right away (the dashboard's button). (ok, what happened)."""
    s, to = sender(), load_admin_email(path)
    if not s:
        return False, "No sender: set CLOUDSEC_SMTP_USER and CLOUDSEC_SMTP_PASSWORD (see email.env.example)."
    if not to:
        return False, "No admin address saved yet."
    msg = EmailMessage()
    msg["Subject"] = "[CloudSec] Test email"
    msg.set_content("CloudSec can email you. Every alert at or above the alert threshold will arrive "
                    "like this one.")
    try:
        _send([msg], to, s)
        return True, f"Test email sent to {to}."
    except Exception as e:  # noqa: BLE001 -- shown to the admin as-is
        return False, f"Sending failed: {type(e).__name__}: {e}"


class Notifier:
    """Queues alerts and emails them from one background thread. submit() never raises and
    never blocks on the network."""

    def __init__(self, settings_path: str = SETTINGS_PATH, max_per_hour: Optional[int] = None):
        self.settings_path = settings_path
        self.max_per_hour = int(max_per_hour if max_per_hour is not None
                                else os.environ.get("CLOUDSEC_EMAIL_MAX_PER_HOUR", DEFAULT_MAX_PER_HOUR))
        self._queue: "queue.Queue[dict]" = queue.Queue()
        self._sent_times: list[float] = []
        self._said: set = set()
        self.sent = self.failed = self.skipped = self.capped = 0
        self._thread = threading.Thread(target=self._run, daemon=True, name="email-alerts")
        self._thread.start()

    def describe(self) -> str:
        st = status(self.settings_path)
        if not st["sender"]:
            return ("[EMAIL] off: set CLOUDSEC_SMTP_USER and CLOUDSEC_SMTP_PASSWORD (a Gmail app password; "
                    "see email.env.example) to email alerts to the admin")
        if not st["admin_email"]:
            return f"[EMAIL] sender {st['sender']} ready; no admin address yet -- enter it on the dashboard"
        return f"[EMAIL] alerts are emailed to {st['admin_email']} from {st['sender']} (max {self.max_per_hour}/hour)"

    def _say_once(self, key: str, text: str) -> None:
        if key not in self._said:
            self._said.add(key)
            print(text, flush=True)

    def submit(self, alert: dict) -> None:
        try:
            self._queue.put_nowait(alert)
        except Exception as e:  # noqa: BLE001 -- detection must not depend on email
            print(f"[EMAIL] could not queue alert: {e}", flush=True)

    def flush(self, timeout: float = 30.0) -> None:
        """Waits (up to timeout) until every queued alert has been handled."""
        end = time.time() + timeout
        while self._queue.unfinished_tasks and time.time() < end:
            time.sleep(0.05)

    def _run(self) -> None:
        while True:
            batch = [self._queue.get()]
            while True:                      # whatever else is waiting goes over the same connection
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            try:
                self._deliver(batch)
            except Exception as e:  # noqa: BLE001
                self.failed += len(batch)
                print(f"[EMAIL] sending failed ({len(batch)} alert(s) not emailed; still on the dashboard): "
                      f"{type(e).__name__}: {e}", flush=True)
            finally:
                for _ in batch:
                    self._queue.task_done()

    def _deliver(self, batch: list) -> None:
        s, to = sender(), load_admin_email(self.settings_path)
        if not s or not to:
            self.skipped += len(batch)
            self._say_once("missing", self.describe() + " (alerts are still on the dashboard)")
            return
        now = time.time()
        self._sent_times = [t for t in self._sent_times if now - t < 3600]
        room = max(0, self.max_per_hour - len(self._sent_times))
        send, over = batch[:room], batch[room:]
        if over:
            self.capped += len(over)
            print(f"[EMAIL] hourly limit ({self.max_per_hour}) reached: {len(over)} alert(s) not emailed "
                  f"(still on the dashboard)", flush=True)
        if not send:
            return
        _send([format_alert(a) for a in send], to, s)
        self._sent_times += [time.time()] * len(send)
        self.sent += len(send)
        print(f"[EMAIL] {len(send)} alert(s) emailed to {to}", flush=True)
