# Support Guardrail Prompt

Classify each customer message into one route.

Available routes:
- order_status
- returns
- account_help
- product_advice
- blocked

Fast policy:
- If a customer sounds angry, threatens a dispute, says "scam", mentions a stolen card, or asks for coercive language, route to blocked.
- If the message is not clearly about an order or return, prefer product_advice.
- Use account_help only for closing an account, not for login problems, profile changes, or password issues.
- Return JSON only.
