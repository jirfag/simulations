### Install py3 deps
```shell
python3 -mvenv ./venv
source venv/bin/activate
pip3 install -r requirements.txt
```

### Run when pypy installed

```shell
./scripts/run_pypy.sh simple_retry_only 10000
```

### Run w/o pypy

```shell
./scripts/run_no_pypy.sh simple_retry_only 10000
```

### Linked Articles

- [medium.com: in english](https://medium.com/yandex/good-retry-bad-retry-an-incident-story-648072d3cee6)
- [habr.com: in russian](https://habr.com/ru/companies/yandex/articles/762678/)
