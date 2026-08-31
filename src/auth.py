"""
auth.py

Deliberately minimal: a single shared admin password protects the
/admin/* API endpoints. This is NOT the user/admin login system described
as future work -- there are no accounts, no per-admin identity, no
session expiry. It exists so the admin UI isn't wide open, and so a real
login system can later replace just this file without touching main.py's
route logic (require_admin is the only integration point).

Login: POST /admin/login {password} -> a token (HMAC of the password with
a server-side secret). The frontend stores it and sends it back as
`Authorization: Bearer <token>` on every /admin/* call.
"""

import hmac
import hashlib


def generate_token(password: str, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), password.encode("utf-8"), hashlib.sha256).hexdigest()


def check_password(password: str, config: dict) -> str | None:
    """Returns a token if the password is correct, else None."""
    expected_password = config["auth"]["admin_password"]
    if not password or not hmac.compare_digest(password, expected_password):
        return None
    return generate_token(expected_password, config["auth"]["token_secret"])


def check_token(token: str, config: dict) -> bool:
    if not token:
        return False
    expected = generate_token(config["auth"]["admin_password"], config["auth"]["token_secret"])
    return hmac.compare_digest(token, expected)
