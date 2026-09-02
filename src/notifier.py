"""
notifier.py

Notifies an admin that a ticket needs human review. Two backends:
  - "console": just prints (default, always works, good for dev/testing)
  - "email": sends a real email via SMTP using config.admin.email settings

Swap in Slack/Teams/webhook here later if needed -- everything else in the
pipeline only depends on notify_admin() being called, not on how.
"""
import os
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv
load_dotenv(override=True)

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

    smtp_password = os.getenv(
        email_cfg.get("smtp_password_env", "SMTP_PASSWORD")
    )

    print("SMTP user:", email_cfg.get("smtp_user"))
    print("SMTP password loaded:", bool(smtp_password))
    print("SMTP password length:", len(smtp_password) if smtp_password else 0)

    if not smtp_password:
        raise RuntimeError("SMTP_PASSWORD was not loaded from environment")

    msg = EmailMessage()
    msg["Subject"] = subject_line
    msg["From"] = email_cfg["from_address"] or email_cfg["smtp_user"]
    msg["To"] = email_cfg["to_address"]
    msg.set_content(body)

    with smtplib.SMTP(
        email_cfg["smtp_host"],
        email_cfg.get("smtp_port", 587)
    ) as smtp:
        smtp.starttls()
        smtp.login(
            email_cfg["smtp_user"],
            smtp_password
        )
        smtp.send_message(msg)

    print("[notifier] Admin email sent successfully.")


def send_to_customer(config: dict, user_email: str, subject: str, answer: str) -> None:
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

    if not user_email:
        print("[notifier] Customer email missing.")
        return

    email_cfg = config["admin"]["email"]

    smtp_password = os.getenv(email_cfg["smtp_password_env"])
    print("SMTP user:", email_cfg["smtp_user"])
    print("SMTP password loaded:", bool(smtp_password))
    print("SMTP password length:", len(smtp_password) if smtp_password else 0)
    msg = EmailMessage()
    msg["Subject"] = f"Re: {subject}"
    msg["From"] = email_cfg["from_address"]
    msg["To"] = user_email
    msg.set_content(answer)

    with smtplib.SMTP(email_cfg["smtp_host"],email_cfg["smtp_port"]) as smtp:

        smtp.starttls()
        smtp.login(email_cfg["smtp_user"],smtp_password)
        smtp.send_message(msg)

    print(f"[notifier] Reply sent to {user_email}")
