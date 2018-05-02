import pytest

from tlapi import run

@pytest.fixture(scope="session", autouse=True)
def setup_app(request):
    print('in setup')
    app = run.create_app()
    app.run(debug=True, port=4999)
    yield app

    def my_own_session_run_at_end():
        print('In my_own_session_run_at_end()')
        #request.addfinalizer(my_own_session_run_at_end)
