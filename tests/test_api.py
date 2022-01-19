import json

def test_hi(client):
    response = client.get('/hi')
    data = json.loads(response.data)
    print(data)

    assert data.get("git") is not None
