# Design files

## Figma

Figma file (created for this project, drop the SVGs in):
https://www.figma.com/design/liWSUcXFWzQ45YZK0KEs9J

The screens themselves are in `figma-screens/` as SVG files with **real text layers**
(exported from the running app with dom-to-svg, fonts Sora + IBM Plex Mono, the same
dusky palette as `src/styles.css`). Figma imports SVG natively, so:

1. Open the Figma file above (or any file).
2. Drag the `.svg` files from `figma-screens/` onto the canvas
   (or `File → Place image / Import`, or `Shift+Ctrl+K`).
3. Every screen lands as a frame with editable text + shapes. Rename the frames
   to match the file names if Figma keeps the generic name.

The Figma MCP integration only has 20 tool calls / month on the Starter plan, which
is why the screens were exported this way instead of being drawn through the API.

| file | screen | role |
|---|---|---|
| `01_login_desktop.svg` | Login — Azure OpenAI credentials + Auth0 | everyone |
| `02_chat_pipeline_desktop.svg` | Policy Chat — pipeline stages while `/ask` runs | every role |
| `03_chat_answer_desktop.svg` | Policy Chat — streamed answer with citations, chips, trace | every role |
| `04_compliance_dashboard_desktop.svg` | Compliance Dashboard | store_manager, compliance_officer, admin |
| `05_reviewer_console_desktop.svg` | Reviewer Console | compliance_officer, legal_reviewer, admin |
| `06_slo_cost_desktop.svg` | SLO & Cost dashboard | compliance_officer, legal_reviewer, admin |
| `07_corpus_admin_desktop.svg` | Corpus Admin | admin |
| `08_rbac_not_allowed_desktop.svg` | RBAC "not allowed" screen (store_associate opening /review) | – |
| `09_login_mobile.svg` | Login at phone width | – |
| `10_chat_answer_mobile.svg` | Policy Chat at phone width | – |

## Screenshots

`screenshots/` has PNG captures of the same screens (plus the escalation → released flow),
taken against a mock backend. Useful for the README / demo slides.

## Miro

Auth0 → JWT → RBAC sequence + role-to-screen routing:
https://miro.com/app/board/uXjVHo7pyn4=/
