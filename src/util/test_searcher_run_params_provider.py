from .searcher_run_params_provider import SearcherRunParamsProviderBuilder, SearcherRunParamsProvider
from src.runners.run_parameters import RunParameters

def test_search_run_params_provider_single():
    provider = (SearcherRunParamsProviderBuilder(lambda: RunParameters('cifar-10'))
        .add_param('test', [1, 2, 3, 4, 5])
        .build())
    print(f"Fetching all provider configurations!")
    cnt = 0
    print(provider)
    for i, x in enumerate(provider):
        cnt += 1
        assert x != None
        print(f"Params {i}: {x.all_params}")
    assert cnt == 5

def test_search_run_params_provider_double():
    provider = (SearcherRunParamsProviderBuilder(lambda: RunParameters('cifar-10'))
        .add_param('test1', [1, 2, 3, 4, 5])
        .add_param('test2', [1, 2, 3, 4, 5])
        .build())

    cnt = 0
    for x in provider:
        cnt += 1
        assert x != None
        print(x.all_params)
    assert cnt == 25

def test_search_run_params_provider_triple():
    provider = (SearcherRunParamsProviderBuilder(lambda: RunParameters('cifar-10'))
        .add_param('test1', [1, 2, 3, 4, 5])
        .add_param('test2', [1, 2, 3, 4, 5])
        .add_param('test3', [x/10 for x in range(5)])
        .build())

    cnt = 0
    for x in provider:
        cnt += 1
        assert x != None
        print(x.all_params)
    assert cnt == 125