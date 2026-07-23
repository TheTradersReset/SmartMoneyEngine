# Email Alerts (SMTP)

Primary notification channel for live paper trading. Telegram remains optional and is **off by default** (often blocked on corporate networks).

## Environment

| Variable | Required when email on | Default | Notes |
|----------|------------------------|---------|-------|
| `SMTP_HOST` | yes | - | e.g. `smtp.gmail.com`, `smtp.office365.com` |
| `SMTP_PORT` | no | `587` | `465` when using SSL |
| `SMTP_USER` | usually | - | Login user (often same as From) |
| `SMTP_PASSWORD` | usually | - | App password / SMTP secret |
| `SMTP_FROM` | yes | - | From address |
| `SMTP_TO` | yes | - | Recipient (or carrier SMS gateway) |
| `SMTP_USE_TLS` | no | `true` | STARTTLS on port 587 |
| `SMTP_USE_SSL` | no | `false` | Implicit SSL (port 465) |
| `LIVE_PAPER_ENABLE_EMAIL` | no | `true` | Master switch |

Env always overrides `config/live_paper/live_paper.yaml`.

## Gmail App Password

1. Enable 2-Step Verification on the Google account.
2. Create an App Password at https://myaccount.google.com/apppasswords
3. Set:

```bash
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=you@gmail.com
SMTP_PASSWORD=xxxx-xxxx-xxxx-xxxx
SMTP_FROM=you@gmail.com
SMTP_TO=you@gmail.com
SMTP_USE_TLS=true
SMTP_USE_SSL=false
```

## Outlook / Office 365

```bash
SMTP_HOST=smtp.office365.com
SMTP_PORT=587
SMTP_USER=you@company.com
SMTP_PASSWORD=your_password_or_app_password
SMTP_FROM=you@company.com
SMTP_TO=you@company.com
SMTP_USE_TLS=true
SMTP_USE_SSL=false
```

Some tenants require modern auth / app passwords; if login fails, ask IT for SMTP relay settings.

## Corporate SMTP

Ask IT for the internal relay (often no password on-network):

```bash
SMTP_HOST=mail.company.internal
SMTP_PORT=25
SMTP_USER=
SMTP_PASSWORD=
SMTP_FROM=smartmoney@company.com
SMTP_TO=you@company.com
SMTP_USE_TLS=false
SMTP_USE_SSL=false
```

Or STARTTLS on 587 with service-account credentials if required.

## Phone push via email-to-SMS (optional)

Many carriers expose an email gateway (`number@txt.att.net`, `number@vtext.com`, etc.). Set `SMTP_TO` to that address to get SMS-like alerts. Carrier gateways are best-effort and may truncate subjects/bodies.

## Alerts

**Signal subject** ? `BUY | NIFTY50 | 10:15`
**Outcome subject** ? `OUTCOME | WIN | BUY`

Bodies include entry/SL/targets (T3=`Runner`), risk, latency, and outcome PnL / R-multiple.

## Test

```bash
python -m src.live_paper.email_test
```

Expect `PASS latency_ms=...` and an inbox message with subject `TEST | SmartMoneyEngine | HH:MM`.

## Troubleshooting

| Symptom | Check |
|---------|--------|
| `FAIL reason=missing_smtp_credentials` | `SMTP_HOST`, `SMTP_FROM`, `SMTP_TO` in `.env` |
| Auth failed / 535 | Wrong password; use Gmail App Password; confirm TLS vs SSL |
| Timeout | Firewall blocks outbound 587/465; try corporate relay |
| Email disabled warning | Host/from/to empty while `LIVE_PAPER_ENABLE_EMAIL=true` |
| No mail but PASS | Spam folder; wrong `SMTP_TO` |
| Port 465 | Set `SMTP_USE_SSL=true`, `SMTP_USE_TLS=false`, `SMTP_PORT=465` |

Logs: `logs/live_paper/email.log` (logger `live_paper.email`).
