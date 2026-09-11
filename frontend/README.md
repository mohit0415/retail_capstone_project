# Retail Policy Intelligence — frontend

React + TypeScript (Vite) UI for the FastAPI backend in `../backend`.
Auth is Auth0 (same method as the Course-2 practice-1 *securing the trip travel planner agent*),
access is role based (RBAC), and the Azure OpenAI credentials are typed on the login page —
they are **not** in any `.env`.

Dusky dark theme, responsive down to phone width, chat answers are shown as a live stream.

## Screens by role

| screen | route | store_associate | store_manager | compliance_officer | legal_reviewer | admin |
|---|---|:-:|:-:|:-:|:-:|:-:|
| Policy Chat (`POST /ask`) | `/chat` | ✓ | ✓ | ✓ | ✓ | ✓ |
| Compliance Dashboard (`/health`, `/requests/{id}`) | `/dashboard` | | ✓ | ✓ | | ✓ |
| Reviewer Console (`/review/*`) | `/review` | | | ✓ | ✓ | ✓ |
| SLO & Cost (`/metrics/slo`, `/metrics/optimization`) | `/slo` | | | ✓ | ✓ | ✓ |
| Corpus Admin (`/ingest*`, `/cache/invalidate`) | `/corpus` | | | | | ✓ |

The backend decides this (`GET /auth/me` returns `screens` + `permissions` for the role, see
`backend/src/auth/rbac.py`), the UI only hides / locks what the role cannot open. Every endpoint is
also protected on the backend, so a locked screen would answer `403` anyway.

## 1. Auth0 setup (once)

Same tenant, application and API as practice-1, so most of this already exists:

1. **API** — Auth0 dashboard → Applications → APIs → `https://api.stateful-agent.com` (RS256).
2. **Application** — the SPA app (client id in `.env`). Allowed Callback URLs, Allowed Logout URLs and
   Allowed Web Origins must contain `http://localhost:5173` (no path — logout returns to the origin).
3. **Roles** — User Management → Roles → create these five (names must match exactly):
   `store_associate`, `store_manager`, `compliance_officer`, `legal_reviewer`, `admin`
   (the old `user` role from practice-1 is still accepted and maps to `store_associate`).
4. **Assign roles** to your test users: User Management → Users → click the user → **Roles** tab →
   **Assign Roles**. Creating a role does not give it to anyone — this step is separate.
   Existing practice-1 users already work: `user` → store_associate, `admin` → admin.
5. **Action** `add-roles-to-tokens` (Actions → Library → Build custom, trigger *Login / Post Login*,
   add it to the Login flow):

```js
exports.onExecutePostLogin = async (event, api) => {
  const namespace = 'https://stateful-agent.com';
  if (event.authorization) {
    api.idToken.setCustomClaim(`${namespace}/roles`, event.authorization.roles);
    api.accessToken.setCustomClaim(`${namespace}/roles`, event.authorization.roles);
  }
  // optional: departments for the department-scoped SQL tables
  const departments = (event.user.app_metadata && event.user.app_metadata.departments) || [];
  api.idToken.setCustomClaim(`${namespace}/departments`, departments);
  api.accessToken.setCustomClaim(`${namespace}/departments`, departments);
};
```

## 2. Configure

`frontend/.env`

```
VITE_AUTH0_DOMAIN=dev-0iuzey7e1lc2ao5e.jp.auth0.com
VITE_AUTH0_CLIENT_ID=SvS5vHl2Wi7ARvUqajZL3G4dqmKpz927
VITE_AUTH0_AUDIENCE=https://api.stateful-agent.com
VITE_AUTH0_ROLE_NAMESPACE=https://stateful-agent.com
VITE_API_URL=http://localhost:8000
VITE_AUTH_MODE=auth0
```

`backend/.env` needs the matching values (see `backend/.env.example`):

```
AUTH0_DOMAIN=dev-0iuzey7e1lc2ao5e.jp.auth0.com
API_AUDIENCE=https://api.stateful-agent.com
ALGORITHMS=RS256
AUTH0_ROLE_NAMESPACE=https://stateful-agent.com
```

`AZURE_OPENAI_ENDPOINT` / `AZURE_OPENAI_API_KEY` can stay empty in the backend `.env` — the login
page supplies them.

## 3. Run

```bash
# backend (from ../backend)
uv sync
uvicorn main:app --reload --port 8000

# frontend
npm install
npm run dev          # http://localhost:5173
```

`npm run build` type-checks and builds to `dist/`, `npm run lint` runs eslint.

## 4. How the login works

```
login page                     Auth0                       backend
──────────                     ─────                       ───────
type Azure endpoint/key/…  ─►  (kept in sessionStorage)
"Continue with Auth0"      ─►  Universal Login  ─►  redirect back with tokens (roles claim inside)
                                                    POST /auth/azure  (Bearer token + the Azure creds)
                                                      └ backend verifies them with 1 tiny embedding
                                                        + 1 tiny chat call, then applies them
                                                    GET  /auth/me     ─► role, scopes, screens
router shows only the screens for the role
```

- `src/main.tsx` wraps the app in `<Auth0Provider>` (domain / clientId / audience).
- `src/hooks/useAuth.tsx` — the `useAuth()` hook: wraps `useAuth0()`, reads the roles claim,
  `getToken()` = `getAccessTokenSilently()`, and talks to `/auth/azure` + `/auth/me`.
- `src/components/RequireAuth.tsx` — not logged in → `/login`; no Azure creds in this tab → `/login`.
- `src/components/RequireScreen.tsx` — screen not in `me.screens` → "not allowed" page.
- `src/apiService.ts` — every call sends `Authorization: Bearer <token>`.

The Azure credentials live in `sessionStorage` only (gone when the tab closes) and are sent once
per session to the backend, which swaps them into its running settings (`backend/src/auth/azure_credentials.py`).

## 5. Chat streaming

`POST /ask` runs the whole graph (guardrails → routing → retrieval → validation → confidence gate)
and returns one certified JSON answer — it cannot stream tokens before the answer is certified.
So the UI does the streaming part itself:

1. while the request is running it animates the pipeline stages,
2. when the answer arrives `streamText()` in `src/apiService.ts` reveals it chunk by chunk
   with a blinking cursor (looks like SSE, no backend change),
3. then the risk / confidence / path chips, citations, SQL proof and decision trace appear.

`202 pending_review` answers are polled on `GET /requests/{id}` every 6 s until a reviewer releases
them; refusals and clarifications get their own cards (clarification candidates are clickable).

## 6. Dev mode without Auth0

`VITE_AUTH_MODE=mock` (+ `VITE_MOCK_ROLE=admin|store_manager|…|none`, `none` = a user without a role) fakes the Auth0 context so the
screens can be opened without a tenant (see `src/auth/devMockAuth.tsx`). The backend still wants a
real token, so this is only for CSS work / screenshots against a mock backend. Never use it in
production.

## 7. Design

Figma file + SVG screens + screenshots: see `design/README.md`.

## 8. Troubleshooting: "no role" screen after login

Login worked and the token is valid, but the **access token** has no usable role. The screen shows what the
ID token and the access token carry and names the fix; the four cases are:

| what the screen shows | cause | fix in Auth0 |
|---|---|---|
| access token roles `[ ] empty` | the user has no role assigned | Users → user → Roles tab → Assign Roles |
| ID token has roles, access token `claim missing` | the Action only calls `api.idToken.setCustomClaim` | add the `api.accessToken.setCustomClaim` line → Deploy |
| both `claim missing` | the Action is not deployed or not in the Login flow | Actions → Triggers → post-login → drag it in → Apply |
| roles under another key | namespace differs | set `AUTH0_ROLE_NAMESPACE` + `VITE_AUTH0_ROLE_NAMESPACE` to it |

Auth0 puts roles into the token **at login**, so after fixing click **I fixed it — refresh token**
(or Sign in again). Quick check that the Action is fine: sign in as a practice-1 user that already has the
`admin` or `user` role — if that works, only the role assignment of the new user was missing.
