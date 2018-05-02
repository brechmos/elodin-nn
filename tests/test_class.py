import pytest
import requests

from tlapi import run

@pytest.fixture(scope="session", autouse=True)
def app(request):
    app = run.create_app()

    ctx = app.app_context()
    ctx.push()

    def teardown():
        ctx.pop()

    request.addfinalizer(teardown)
    return app


def test_one(app):
    print(app)
    resp = requests.get('http://localhost:5000/data')
    print(resp.data)

    x = "this"
    assert 'k' in x

