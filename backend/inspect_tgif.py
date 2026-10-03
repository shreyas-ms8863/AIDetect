import urllib.request

candidates = [
    'masks', 'original', 'orig', 'Original', 'SD2-sp', 'sd2-sp', 'PS-sp', 'ps-sp',
    'SD2-fr', 'sd2-fr', 'SDXL-fr', 'sdxl-fr', 'flux1schnell-sp', 'flux1dev-sp',
    'metadata', 'code'
]

for name in candidates:
    url = f'https://cloud.ilabt.imec.be/index.php/s/xEeAzrY7ES9KA8o/download?files={name}'
    req = urllib.request.Request(url, method='HEAD', headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            print(f"MATCH: {name} -> {resp.status}, disp: {resp.headers.get('Content-Disposition')}")
    except Exception as e:
        pass

