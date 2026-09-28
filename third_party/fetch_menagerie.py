# Re-fetch menagerie franka_fr3 (57 MB, not committed) at a pinned commit.
import json, os, urllib.request
from concurrent.futures import ThreadPoolExecutor
SHA = 'c96a32d28fb5da84da38c1da4d749e7a13212855'
ROOT = os.path.join(os.path.dirname(__file__), 'menagerie')
def get(u):
    return urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent': 'curl/8'}), timeout=60).read()
tree = json.loads(get(f'https://api.github.com/repos/google-deepmind/mujoco_menagerie/git/trees/{SHA}?recursive=1'))['tree']
files = [x['path'] for x in tree if x['path'].startswith('franka_fr3/') and x['type'] == 'blob']
def dl(p):
    out = os.path.join(ROOT, p)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, 'wb').write(get(f'https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{SHA}/{p}'))
with ThreadPoolExecutor(6) as ex:
    list(ex.map(dl, files))
print(len(files), 'files ->', ROOT)
