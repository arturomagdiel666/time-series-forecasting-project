# Deployment checklist

Manual steps for the repository owner. These are executed by **you**, not by any automated tooling — they involve pushing to GitHub and deploying to Railway. Follow them in order.

Repository: `arturomagdiel666/time-series-forecasting-project` · default branch `main`.

## 1. Confirm what is (and is not) committed

Railway builds from the GitHub repo and has **no access to the raw dataset**, so the processed artifacts must be committed while the raw file must not.

`.gitignore` must **exclude**: `data/raw/*.txt`, `mlruns/`, `.venv/`, `__pycache__/`.
The repo must **keep** (already committed): `data/processed/*.parquet`, `data/processed/predictions/*.parquet`, `models/*.pkl`, `models/metadata.json`, and `reports/figures/`.

Verify:

```bash
git check-ignore data/raw/household_power_consumption.txt   # should print the path (ignored)
git ls-files models/ data/processed/                        # should list the parquets and .pkl/.json
```

## 2. Create the GitHub repo and push

```bash
gh repo create arturomagdiel666/time-series-forecasting-project --public --source=. --remote=origin
# or, without the gh CLI, create the empty repo on github.com then:
#   git remote add origin https://github.com/arturomagdiel666/time-series-forecasting-project.git

git push -u origin main
```

## 3. Confirm CI is green

Open the Actions tab and confirm the `CI` workflow run passes:

`https://github.com/arturomagdiel666/time-series-forecasting-project/actions/workflows/ci.yml`

The README CI badge turns green once the run succeeds.

## 4. Deploy on Railway

1. Railway → **New Project** → **Deploy from GitHub repo** → select `time-series-forecasting-project`.
2. Railway detects the `Procfile`. Confirm the start command binds the injected port and all interfaces:

   ```
   web: streamlit run app/dashboard.py --server.port $PORT --server.address 0.0.0.0
   ```

3. Deploy and wait for the build to finish. The build installs `requirements.txt`; no extra config is required because the dashboard reads only committed artifacts.

## 5. Publish the live URL

Copy the Railway URL into the README placeholder and push the one-line update:

```bash
# edit README.md: replace  <RAILWAY_URL — add after deploy>  with the real URL
git add README.md
git commit -m "docs: add live dashboard url"
git push
```

## 6. Smoke-test the live app

Open the Railway URL and confirm:

- All 7 pages load from the sidebar (Home, Exploration, Statistical Analysis, Model Comparison, Forecasting Playground, Scientific Report, Video Explanation).
- No traceback appears; charts render on the Exploration and Model Comparison pages.
- The Forecasting Playground returns next-hour and next-day forecasts for a selected timestamp.

Deployment is complete once the live app passes this smoke test.
