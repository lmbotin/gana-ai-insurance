## Rex AI insurance platform

CS224G Project: Luis Botin (lmbotin@stanford.edu) and Jesus Santos (jsaranda@stanford.edu)

AI-native insurance platform for operational/ AI liability claims.

---

**<ins>Current demo status (sprint 2)<ins>**

Easiest way to see the demo today is

1. Go to https://rex-ai-insurance.vercel.app
2. Click Sign Up and create an account (any email/password. Stored in browser localStorage.)
3. Once inside the dashboard you'll see:

- Overview: mock UI, hardcoded dashboard metrics and charts
- Policies: mock UI, local only policy creation flow (no backend)
- Claims > Chat with Sarah: live AI, connected to the real FastAPI backend on Render and powered by GPT-4o
- Rex AI (Copilot): hardcoded pattern matched responses, not connected to any LLM for the moment (the plan is for Rexy to be the general, all purpose agent connecting all parts of the platform)

The core demo right now is "Chat with Sarah" inside the Claims page. This is the only feature currently connected to the actual backend AI agent (hosted on Render). Sarah walks you through filing an operational liability claim (verifies policy, collects incident details, processes claim through the routing pipeline...). On the right you see all extracted data, and a claim ID is created. 

Note: the policy needs to be seeded in the backend. So for the demo, you need to use one of the companies currently in data/policies/ai_logistics_policies.json. For example use "Beta Freight Co." with policy number 555123. 

Example conversation:

You: Hi I need to file a claim for Beta Freight Co, policy 555123

Sarah:  Got it, thanks. I've verified Beta Freight Co. under policy 555123 — everything matches our records. What type of incident was it? You can describe it in your own words (e.g. pricing error, delay, misroute, loss, system outage).

You: It was a pricing error. Our system quoted a customer $3,000 but the actual cost of the load was $10,000

... 

[continue until all relevant info has been extracted]

Everything else (Overview, Policies, Rex AI copilot) is placeholder UI to show the intended product direction

---

**<ins>Running locally<ins>** (same as above, just locally, without render/vercel)

Prerequisites:
- Node.js >= 18 (tested with v20+)
- npm (comes with Node.js)

**Frontend only (mock UI, no AI chat)**

This gets the full UI running locally. The dashboard, policies, claims pages all work with browser-local data. The "Chat with Sarah" feature will not work without backend

```bash
# 1. clone repo
git clone https://github.com/<your-org>/Rex.git
cd Rex/frontend

# 2. install dependencies
npm install

# 3. start the dev server
npm run dev
```

The app will be available at http://localhost:5173.

To also run the local Node.js API server (handles call requests only):

```bash
# In a second terminal
cd Rex/frontend
npm run api
```

This starts lightweight server on port 8787 that the Vite dev server proxies to

**Frontend + Backend (full AI chat)**

To get the "Chat with Sarah" AI working locally, you also need to run the python backend.

Backend setup:

```bash
cd Rex/backend

# create a virtual environment
python3 -m venv venv
source venv/bin/activate

# install dependencies
pip install -r requirements.txt

# set up environment variables
cp ../.env.example .env
# Edit .env and add your OpenAI API key:
#   OPENAI_API_KEY=sk-your-key-here
```

Start the backend:

```bash
cd Rex/backend
source venv/bin/activate
uvicorn src.web.chat_app:app --host 0.0.0.0 --port 8000
```

The backend API will be available at http://localhost:8000.

Start the frontend (pointed at local backend):

The frontend `chatApi` defaults to `http://localhost:8000` in development so just run:

```bash
cd Rex/frontend
npm run dev
```

Then go to http://localhost:5173, sign up, navigate to Claims > Chat with Sarah and start a conversation

---

**<ins>Project structure<ins>**

```
Rex/
├── backend/                 # python FastAPI backend (deployed on Render)
│   ├── src/
│   │   ├── web/chat_app.py  # Chat API (the core AI agent)
│   │   ├── fnol/            # claim extraction pipeline (text + images)
│   │   ├── policy/          # Policy verification service
│   │   ├── routing/         # Claims routing and workflow
│   │   ├── voice/           # Twilio voice agent (separate feature)
│   │   └── utils/           # Config and utilities
│   ├── requirements.txt
│   ├── Procfile             # Render deployment
│   └── runtime.txt          # Python 3.11
│
├── frontend/                # React+ Vite frontend (deployed on Vercel)
│   ├── src/
│   │   ├── pages/dashboard/ # Overview, Claims, Policies, Copilot, etc
│   │   ├── components/      # ChatInterface, UI components (shadcn/ui)
│   │   ├── hooks/useChat.js # Chat state management hook
│   │   ├── lib/             # Utilities, API clients, storage
│   │   ├── context/         # App-wide state (auth, policies, claims)
│   │   └── layouts/         # Dashboard shell
│   ├── server/              # Lightweight Node.js dev API (port 8787)
│   ├── vite.config.js
│   └── package.json
│
├── data/                    # Sample claims, policy schemas, examples
├── .env.example             # Environment variable template
└── README.md
```

**<ins>Tech stack<ins>**

- Frontend — React 19, Vite, Tailwind CSS, Radix UI, Recharts
- Backend — Python 3.11, FastAPI, Uvicorn
- AI — OpenAI GPT-4o (chat + vision + extraction)
- Deployment — Vercel (frontend), Render (backend)

**<ins>Environment vars<ins>**

The backend requires an OpenAI API key. Copy `.env.example` to `backend/.env` and set:

```
OPENAI_API_KEY=sk-your-key-here
```

All other variables are optional with any reasonble defaults. See `.env.example` for full list (Twilio keys are only needed for voice agent feature).

---

**<ins>TDeployment<ins>**

- Frontend: Vercel — root directory: `frontend/`, build command: `npm run build`, output: `dist/`
- Backend: Render — root directory: `backend/`, start command: `uvicorn src.web.chat_app:app --host 0.0.0.0 --port $PORT`
