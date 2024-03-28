# ese5934_project


## BART Installation
To install BART, please refer to https://mrirecon.github.io/bart/ . 
You download their release and then run `make` to compile the code. For redhat based system and mac os I have tested that it works. For windows, you may need to use WSL or other linux environment to compile the code.

## Denpendency Installation
Too install all the dependency, we recommend using `poetry` as dependency manager.
### Peotry Installation
Run following code to install `poetry`
```bash
curl -sSL https://install.python-poetry.org | python3 -
```
then add `$HOME/.local/bin` to your PATH (example code is given for bash, you may need to find out how to do it for your own shell):
```bash
echo 'export PATH=$PATH:$HOME/.local/bin' >> ~/.bashrc
```
and then open a new shell and try 
```bash
poetry --version

```
if it return something like `Poetry (version 1.2.0)`, your install is ready.

Then just run in this repo's folder
```bash
poetry install
```
it will automatically install all the dependency.

### Poetry Add Dependency
For example if you want to add `nibabel` to the denpendency just run
```bash
poetry add nibabel
```
### Poetry Detailed
For more information please refer to https://python-poetry.org/docs/


testing change by mahshid