# ese5934_project
Too install all the dependency, We recommend using `poetry` as dependency manager.

Run following code to install `poetry`
```bash
curl -sSL https://install.python-poetry.org | python3 -
```
then add `$HOME/.local/bin` to your PATH (example code is given for bash, you may need to find out how to do it for your own shell):
```bash
echo 'export PATH=$PATH:$HOME/.local/bin' >> ~/.bashrc
```
and then try 
```bash
poetry --version

```
if it return something like `Poetry (version 1.2.0)`, your install is ready.

Then just run in this repo's folder
```bash
poetry install
```
it will automatically install all the dependency.
