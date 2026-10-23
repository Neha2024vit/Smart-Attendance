from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    r = client.get('/api/health')
    assert r.status_code == 200
    assert r.json()['status'] == 'ok'

def test_home_contains_disclaimer():
    r = client.get('/')
    assert r.status_code == 200
    assert 'not production-grade security' in r.text

def test_challenge():
    r = client.get('/api/challenge')
    assert r.status_code == 200
    assert r.json()['challenge']
