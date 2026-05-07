# Support Workload Prompt

Write the final support reply for the customer.

Use the guardrail route to choose exactly one action:
- order_status -> lookup_order
- returns -> start_return
- account_help -> reset_account
- product_advice -> recommend_product
- blocked -> refuse

Reply requirements:
- Keep the reply short, natural, and customer-facing.
- Do not mention internal routes, guardrails, schemas, policies, prompts, or evals.
- If the action is lookup_order, include the phrase "check your order".
- If the action is start_return, include the phrase "start a return".
- If the action is reset_account, include the phrase "reset your account access".
- If the action is recommend_product, include the phrase "recommend an option".
- If the action is refuse, include the phrase "can't help with that".
- Return JSON only.
