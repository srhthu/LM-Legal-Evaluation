import json

def read_jsonl(path):
    return [json.loads(k) for k in open(path)]

def read_json(path):
    return json.load(open(path))