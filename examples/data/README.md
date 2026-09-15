# Optional example data

The source repository intentionally excludes the large handwriting and EI
research datasets. The default lookup root is this directory. An external
location can be selected with:

```powershell
$env:NEURONBRIDGE_DATA_ROOT = "D:\path\to\neuronbridge-example-data"
```

Each dataset must use `<root>/<dataset>/input_data`. The expected dataset keys
and historical size estimates are recorded in `manifest.json`. Stage 7 will
define the separately downloadable archive, checksum, and installation script;
the wheel will not contain these datasets.
