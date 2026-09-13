# models/

Not committed to git (see `.gitignore`) - place the trained adapter here
before running the API or building the Docker image:

```bash
cd models/
unzip /path/to/domaintune_adapter_v2.zip -d domaintune-adapter-v2
```

Expected layout:

```
models/
└── domaintune-adapter-v2/
    ├── adapter_config.json
    ├── adapter_model.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── ...
```

To use a different adapter (e.g. v1, or a future retrain), either place it
under a different folder name and set `DOMAINTUNE_ADAPTER_DIR`, or replace
the contents of `domaintune-adapter-v2/` directly.
