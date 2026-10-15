"""
test_notify.py
==============
Email alerts (notify.py): an alert reaches the admin when everything is configured, and detection
is never affected when it isn't -- missing sender, missing address, a mail server that fails, or
the hourly cap. No network: smtplib is replaced by a fake server that records what it was sent.
"""

import os
import tempfile
import unittest
from unittest import mock

import notify


def _alert(risk=8.43, principal="stratus-redteam"):
    return {
        "alert_id": "a1", "source_file": "batch0001.csv", "principal": principal,
        "principal_arn": "arn:aws:iam::123456789012:user/stratus-redteam",
        "first_seen": "2026-10-14 10:00:00+00:00", "last_seen": "2026-10-14 10:02:00+00:00",
        "max_risk_score": risk, "n_flagged_events": 3, "fast_lane_reasons": [], "threshold": 0.42,
        "events": [{"log_id": "x:1", "timestamp": "2026-10-14 10:01:00+00:00", "event_name": "PutRolePolicy",
                    "target": "role/app", "p_graph": 0.86, "p_sequence": 0.82, "risk_score": risk}],
        "explanations": [{"summary": "risk 8.43/10 | LSTM 49%: the action PutRolePolicy"}],
    }


class _FakeSMTP:
    instances = []
    fail_login = False

    def __init__(self, host, port, timeout=None, context=None):
        self.host, self.port, self.sent, self.logged_in = host, port, [], None
        _FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def login(self, user, password):
        if _FakeSMTP.fail_login:
            raise OSError("535 bad credentials")
        self.logged_in = (user, password)

    def send_message(self, msg):
        self.sent.append(msg)


class NotifyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.settings = os.path.join(self.tmp.name, "notify_settings.json")
        _FakeSMTP.instances, _FakeSMTP.fail_login = [], False
        self.env = mock.patch.dict(os.environ, {"CLOUDSEC_SMTP_USER": "sender@gmail.com",
                                                "CLOUDSEC_SMTP_PASSWORD": "abcd efgh ijkl mnop"}, clear=False)
        self.env.start()
        os.environ.pop("CLOUDSEC_ADMIN_EMAIL", None)
        self.smtp = mock.patch("notify.smtplib.SMTP_SSL", _FakeSMTP)
        self.smtp.start()

    def tearDown(self):
        self.smtp.stop()
        self.env.stop()
        self.tmp.cleanup()

    def sent(self):
        return [m for s in _FakeSMTP.instances for m in s.sent]

    def test_alert_is_emailed_to_the_saved_admin_address(self):
        notify.save_admin_email("admin@example.com", self.settings)
        n = notify.Notifier(self.settings)
        n.submit(_alert())
        n.flush(5)
        msgs = self.sent()
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["To"], "admin@example.com")
        self.assertEqual(msgs[0]["From"], "sender@gmail.com")
        self.assertEqual(_FakeSMTP.instances[0].logged_in, ("sender@gmail.com", "abcdefghijklmnop"))
        self.assertEqual((_FakeSMTP.instances[0].host, _FakeSMTP.instances[0].port), ("smtp.gmail.com", 465))

    def test_email_says_who_how_risky_and_why(self):
        body = notify.format_alert(_alert()).get_content()
        subject = notify.format_alert(_alert())["Subject"]
        self.assertIn("8.43/10", subject)
        self.assertIn("stratus-redteam", subject)
        for text in ("PutRolePolicy", "alerts at 4.20/10", "Why it was flagged", "LSTM 49%"):
            self.assertIn(text, body)

    def test_nothing_configured_sends_nothing_and_raises_nothing(self):
        with mock.patch.dict(os.environ, {"CLOUDSEC_SMTP_USER": "", "CLOUDSEC_SMTP_PASSWORD": ""}):
            n = notify.Notifier(self.settings)
            n.submit(_alert())
            n.flush(5)
            self.assertEqual(self.sent(), [])
            self.assertEqual(n.skipped, 1)
            self.assertIn("off", n.describe())

    def test_no_admin_address_sends_nothing(self):
        n = notify.Notifier(self.settings)
        n.submit(_alert())
        n.flush(5)
        self.assertEqual(self.sent(), [])
        self.assertEqual(n.skipped, 1)

    def test_failing_mail_server_does_not_raise(self):
        notify.save_admin_email("admin@example.com", self.settings)
        _FakeSMTP.fail_login = True
        n = notify.Notifier(self.settings)
        n.submit(_alert())
        n.flush(5)
        self.assertEqual(n.failed, 1)
        self.assertEqual(n.sent, 0)

    def test_hourly_cap(self):
        notify.save_admin_email("admin@example.com", self.settings)
        n = notify.Notifier(self.settings, max_per_hour=2)
        for i in range(5):
            n.submit(_alert(principal=f"user{i}"))
        n.flush(5)
        self.assertEqual(n.sent, 2)
        self.assertEqual(n.capped, 3)

    def test_address_change_applies_without_restart(self):
        notify.save_admin_email("first@example.com", self.settings)
        n = notify.Notifier(self.settings)
        n.submit(_alert())
        n.flush(5)
        notify.save_admin_email("second@example.com", self.settings)
        n.submit(_alert())
        n.flush(5)
        self.assertEqual([m["To"] for m in self.sent()], ["first@example.com", "second@example.com"])

    def test_saved_address_beats_environment_and_can_be_cleared(self):
        with mock.patch.dict(os.environ, {"CLOUDSEC_ADMIN_EMAIL": "env@example.com"}):
            self.assertEqual(notify.load_admin_email(self.settings), "env@example.com")
            notify.save_admin_email("dash@example.com", self.settings)
            self.assertEqual(notify.load_admin_email(self.settings), "dash@example.com")
            notify.save_admin_email("", self.settings)
            self.assertEqual(notify.load_admin_email(self.settings), "env@example.com")

    def test_bad_address_is_refused(self):
        for bad in ("admin", "admin@", "a b@example.com", "@example.com"):
            with self.assertRaises(ValueError):
                notify.save_admin_email(bad, self.settings)
        self.assertFalse(os.path.exists(self.settings))


if __name__ == "__main__":
    unittest.main()
