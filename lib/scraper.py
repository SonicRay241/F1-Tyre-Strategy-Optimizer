import requests
import os
from urllib.parse import urlencode
import json
import hashlib

def get_json(url, params={}, cache_path="../data/cache"):
    """Helper to GET JSON safely. Also comes with caching."""
    str_id = f"{url}-{urlencode(params)}".encode("utf-8")

    hasher = hashlib.sha256()
    hasher.update(str_id)
    url_hash = hasher.hexdigest()

    try:
        with open(os.path.join(cache_path, f"{url_hash}.json"), "r") as cache:
            content =  cache.read()
            return json.loads(content)
    
    except:
        r = requests.get(url, params=params)
        r.raise_for_status()
        result = r.json()

        # Make sure the directory exists
        os.makedirs(cache_path, exist_ok = True)

        try:
            with open(os.path.join(cache_path, f"{url_hash}.json"), "w") as new_cache:
                new_cache.write(json.dumps(result))
        
        except:
            raise Exception("Error when writing cache!")

    return result