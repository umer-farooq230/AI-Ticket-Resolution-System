"""
notifier.py

Notifies an admin that a ticket needs human review. Two backends:
  - "console": just prints (default, always works, good for dev/testing)
  - "email": sends a real email via SMTP using config.admin.email settings

Swap in Slack/Teams/webhook here later if needed -- everything else in the
pipeline only depends on notify_admin() being called, not on how.
"""

import smtplib
from email.message import EmailMessage


def notify_admin(config: dict, pending_id: int, subject: str, body: str,
                  draft_answer: str, confidence: float, retrieval_score: float) -> None:
    method = config["admin"]["notify_method"]
    message = (
        f"[Pending review #{pending_id}] confidence={confidence:.2f} "
        f"retrieval_score={retrieval_score:.2f}\n"
        f"Subject: {subject}\n"
        f"Customer message: {body}\n\n"
        f"Draft answer:\n{draft_answer}\n"
    )

    if method == "email":
        _send_email(config, subject_line=f"[Review needed] {subject}", body=message)
    else:
        print("=" * 60)
        print("ADMIN NOTIFICATION (console)")
        print("=" * 60)
        print(message)
        print("=" * 60)


def _send_email(config: dict, subject_line: str, body: str) -> None:
    email_cfg = config["admin"]["email"]
    if not email_cfg.get("smtp_host") or not email_cfg.get("to_address"):
        print("[notifier] email notify_method configured but smtp_host/to_address "
              "missing -- falling back to console output.")
        print(body)
        return

    msg = EmailMessage()
    msg["Subject"] = subject_line
    msg["From"] = email_cfg["from_address"] or email_cfg["smtp_user"]
    msg["To"] = email_cfg["to_address"]
    msg.set_content(body)

    with smtplib.SMTP(email_cfg["smtp_host"], email_cfg.get("smtp_port", 587)) as smtp:
        smtp.starttls()
        if email_cfg.get("smtp_user"):
            smtp.login(email_cfg["smtp_user"], email_cfg.get("smtp_password", ""))
        smtp.send_message(msg)


def send_to_customer(user_email: str, subject: str, answer: str) -> None:
    """
    Stub for the final "send the answer to the customer" step, used both
    for auto-sent answers and admin-approved drafts. Wire this up to your
    real email/helpdesk API -- left as a console print here.
    """
    print("=" * 60)
    print(f"SENDING TO CUSTOMER ({user_email or 'unknown address'})")
    print("=" * 60)
    print(f"Subject: Re: {subject}\n\n{answer}")
    print("=" * 60)
