# Translate pixi.lock into files that can be used with conda create --file

import yaml

with open('pixi.lock') as infile:
    d = yaml.safe_load(infile.read())
    for arch, packages in d['environments']['default']['packages'].items():
        urls = [p['conda'] for p in packages]
        with open(f'conda-{arch}.lock', 'w') as outfile:
            print('@EXPLICIT', file=outfile)
            for url in urls:
                print(url, file=outfile)
