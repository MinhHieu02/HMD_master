import yaml

with open('environment.yml') as f:
    env = yaml.safe_load(f)

with open('requirements.txt', 'w') as f:
    for dep in env['dependencies']:
        if isinstance(dep, str):  # conda dependency
            f.write(dep.split('=')[0] + '==' + dep.split('=')[1] + '\n')
        elif isinstance(dep, dict):  # pip dependency
            for pip_dep in dep['pip']:
                f.write(pip_dep + '\n')