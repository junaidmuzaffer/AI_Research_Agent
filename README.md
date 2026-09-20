# AI Research Agent — GitHub to Streamlit

A beginner-friendly, single-agent CrewAI app using Groq `openai/gpt-oss-120b` and DuckDuckGo search. No local installation is needed. Search results are snippets, not full articles. This is a research starting point, not a full-page research or document-RAG system.

## Files to put in the repository root

| File | Purpose |
| --- | --- |
| app.py | Interface, search tool, single research agent and report download |
| requirements.txt | Pinned direct dependencies |
| README.md | These instructions |
| .gitignore | Excludes secrets and local temporary files |

## Deploy using only your browser

1. Download and extract the ZIP on your computer. Do not upload the ZIP itself to GitHub.
2. Sign in at https://github.com and select New repository. Name it `ai-research-agent`. Choose your preferred visibility and create it.
3. Choose **Add file → Upload files** (or **uploading an existing file** in an empty repository). Upload the extracted files so `app.py` and `requirements.txt` are directly at the repository root. Commit changes.
4. Sign in at https://console.groq.com/keys and create an API key. Keep it for Step 7. You do not need an OpenAI API key.
5. Open https://share.streamlit.io and sign in with GitHub. Allow access to your selected repository.
6. Choose **Create app**, deploy from GitHub, select your repository and `main` branch, and set the main file path to `app.py`.
7. Open **Advanced settings**. Choose **Python 3.11** and paste this into **Secrets**, replacing the placeholder:

```toml
GROQ_API_KEY = "paste_your_actual_groq_key_here"
```

8. Click **Deploy**. Dependencies install on Streamlit's servers; CrewAI has a sizeable dependency tree, so initial installation can take several minutes.
9. Enter a short research topic and click **Generate report**. Try: `Applications of AI in engineering project planning`.
10. Review the report and source evidence. Click **Download report (.md)**. A Markdown file is plain text you can open in an editor or paste into Word.

Never put your real key in app.py, README.md, GitHub or a committed secrets file. For an already-deployed app, edit **Settings → Secrets**, save and reboot. Private GitHub source does not itself determine who can visit your deployed app; configure app sharing separately.

## How it works

The app retrieves an initial DuckDuckGo search, stops if no usable results are returned, and passes the evidence to one CrewAI research agent. The agent can run two more focused searches and writes a report through Groq. The app appends the actual retrieved source list and retains the report in the current Streamlit session so download clicks do not rerun research. Refreshing or losing the session can clear the report.

`groq/openai/gpt-oss-120b` is the CrewAI/LiteLLM provider-prefixed model name. The actual Groq model ID is `openai/gpt-oss-120b`. Delegation, memory and verbose agent output are disabled. No OpenAI API calls are intended.

## Troubleshooting

| Problem | Action |
| --- | --- |
| Missing key message | Use exact spelling GROQ_API_KEY, with quotes around the value in Secrets. Save and reboot. |
| Groq authentication failure | Generate a valid key and replace the old one in Secrets. |
| Rate limit | Wait and retry; select 400 words. Free-tier limits depend on your Groq account and model. |
| No search results | Try a narrower topic or retry later. DuckDuckGo can restrict requests from shared cloud IP addresses; this app cannot guarantee search availability. |
| Model unavailable | Check GPT-OSS 120B access in your Groq console. |
| Module not found | Confirm requirements.txt is beside app.py and inspect the Streamlit build log. |
| Build failure | Verify Python 3.11 and the exact requirements; share the error text without secrets. |
| App appears asleep | Open the app and allow Streamlit to wake it. |

## Validation and limitations

Direct dependency versions and API usage were checked against official package pages and documentation. Python syntax and the ZIP structure were checked. A full dependency installation and live Groq/Streamlit run were not completed in the preparation environment; network access was restricted and no API key was supplied. These are prepared deployment files, not a claim of a tested live deployment. Transitive dependencies are not fully locked.

Search snippets can be stale or incomplete; source presence does not prove every generated claim. The model is instructed to cite retrieved IDs, but semantic citation correctness is not automatically verified. Always review important claims. Queries go to the search provider; the topic and retrieved snippets go to Groq. No search API key is required; Groq usage is subject to account limits and pricing.

## Official references

- CrewAI model configuration: https://docs.crewai.com/en/learn/llm-connections
- CrewAI custom tools: https://docs.crewai.com/en/learn/create-custom-tools
- Groq model: https://console.groq.com/docs/model/openai/gpt-oss-120b
- DDGS search API: https://pypi.org/project/ddgs/
- Streamlit deployment: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app
