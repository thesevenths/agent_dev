import requests

api_key = "sk-BC14EBvSUyrx6B9M1jOdwEOLOywv2He4VA5cKb7CRsx7pvky"

url = "https://api.openai-proxy.org/v1"
headers = {
    "Authorization": f"Bearer {api_key}"
}

resp = requests.get(url, headers=headers)

print("Status:", resp.status_code)
print("Response:")
print(resp.text)
