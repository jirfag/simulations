### Install py3 deps
```shell
python3 -mvenv ./venv
source venv/bin/activate
pip3 install -r requirements.txt
```

### Run w/o pypy

```shell
python3 simulation_main.py --target-rps 10000 --save-to __by_time.res && python3 simulation_main.py --visualize --load-from __by_time.res && open out.png
```

Run w/o saving and loading:

```shell
python3 simulation_main.py --target-rps 10000 --visualize && open out.png
```

### Run when pypy installed

```shell
./venv-pypy/bin/python3 simulation_main.py --save-to __by_time.res && ./venv/bin/python3 simulation_main.py --visualize --load-from __by_time.res && open out.png
```