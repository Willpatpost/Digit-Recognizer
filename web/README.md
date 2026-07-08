# Browser Demo

This folder contains the static GitHub Pages version of the digit recognizer.
Inference runs entirely in the browser; no Python server or external API is
required after the model has been exported.

## Preview locally

From the repository root:

```bash
python -m http.server 8000 --directory web
```

Open `http://localhost:8000`.

## Replace the trained model

Save a model from the desktop application, then export it:

```bash
python web/export_model.py path/to/model.npz
```

The command replaces `web/models/digit-model.json`, which is loaded automatically
by the webpage.

## Publish with GitHub Pages

The repository includes `.github/workflows/pages.yml`, which publishes this
folder without moving it to the repository root.

1. Push the repository to GitHub.
2. Open **Settings > Pages**.
3. Under **Build and deployment**, select **GitHub Actions** as the source.
4. Run the workflow manually or push a change inside `web/`.

The deployed site will be available at:

`https://<username>.github.io/<repository>/`
