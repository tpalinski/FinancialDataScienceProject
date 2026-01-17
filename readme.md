# CLI

Simulation CLI

**Usage**:

```console
$ [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.

**Commands**:

* `simulate-single`
* `simulate-multidata`
* `simulate-finetuned`

## `simulate-single`

**Usage**:

```console
$ simulate-single [OPTIONS]
```

**Options**:

* `--index [ETH|BTC|XRP|LTC|XLM]`: [default: ETH]
* `--output-shap / --no-output-shap`: [default: no-output-shap]
* `--output-visualize / --no-output-visualize`: [default: no-output-visualize]
* `--test-days INTEGER`: [default: 200]
* `--largest-window INTEGER`: [default: 50]
* `--validation-days INTEGER`: [default: 100]
* `--epochs INTEGER`: [default: 300]
* `--buy-prob FLOAT`: [default: 0.5]
* `--sell-prob FLOAT`: [default: 0.6]
* `--help`: Show this message and exit.

## `simulate-multidata`

**Usage**:

```console
$ simulate-multidata [OPTIONS]
```

**Options**:

* `--index [ETH|BTC|XRP|LTC|XLM]`: [default: ETH]
* `--validation-days INTEGER`: [default: 100]
* `--test-days INTEGER`: [default: 200]
* `--largest-window INTEGER`: [default: 50]
* `--epochs INTEGER`: [default: 300]
* `--buy-prob FLOAT`: [default: 0.5]
* `--sell-prob FLOAT`: [default: 0.6]
* `--help`: Show this message and exit.

## `simulate-finetuned`

**Usage**:

```console
$ simulate-finetuned [OPTIONS]
```

**Options**:

* `--index [ETH|BTC|XRP|LTC|XLM]`: [default: ETH]
* `--output-shap / --no-output-shap`: [default: no-output-shap]
* `--validation-days INTEGER`: [default: 100]
* `--test-days INTEGER`: [default: 200]
* `--largest-window INTEGER`: [default: 50]
* `--epochs INTEGER`: [default: 300]
* `--buy-prob FLOAT`: [default: 0.5]
* `--sell-prob FLOAT`: [default: 0.6]
* `--fine-tune-epochs TEXT`: [default: 100]
* `--help`: Show this message and exit.
