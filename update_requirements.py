import subprocess

def install_or_update_packages(requirements_file: str):
    with open(requirements_file, 'r') as file:
        packages = file.readlines()

    for package in packages:
        package_name = package.strip()
        if package_name:
            subprocess.run(['pip', 'install', '--upgrade', package_name])

# Example usage
install_or_update_packages('src/requirements.txt')