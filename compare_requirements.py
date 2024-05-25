import subprocess

def read_requirements(file_path):
    """Reads a requirements.txt file and returns a set of package names."""
    with open(file_path, 'r') as file:
        return {line.strip().split('=')[0] for line in file if not line.startswith('#')}

def get_installed_packages():
    """Runs 'pip freeze' command and returns a set of installed package names."""
    output = subprocess.check_output(['pip', 'freeze']).decode('utf-8')
    return {line.split('==')[0] for line in output.split('\n')}

def compare_requirements_with_installed(requirements_file):
    """Compares the entries of a requirements.txt file with installed packages."""
    required_packages = read_requirements(requirements_file)
    installed_packages = get_installed_packages()

    missing_packages = required_packages - installed_packages
    extra_packages = installed_packages - required_packages

    if missing_packages:
        print("Missing packages:")
        for package in missing_packages:
            print(f" - {package}")

    if extra_packages:
        print("\nExtra installed packages not in requirements.txt:")
        for package in extra_packages:
            print(f" - {package}")

    if not missing_packages and not extra_packages:
        print("All packages in requirements.txt are installed.")


def install_missing_packages(requirements_file):
    # Read the requirements.txt file
    with open(requirements_file, 'r') as file:
        requirements = file.read().splitlines()

    # Get the list of installed packages using pip freeze
    installed_packages = subprocess.run(['pip', 'freeze'], capture_output=True, text=True)
    installed_packages = installed_packages.stdout.splitlines()

    # Find missing packages
    missing_packages = set(requirements) - set(installed_packages)

    # Prompt user to confirm installation
    if missing_packages:
        print("The following packages need to be installed:")
        for package in missing_packages:
            print(package)
        
        # Ask user for confirmation
        confirmation = input("Do you want to install these missing packages? (yes/no): ").lower()

        if confirmation == 'yes':
            print("Installing missing packages:")
            for package in missing_packages:
                print(package)
                subprocess.run(['pip', 'install', package])
        else:
            print("Installation cancelled.")
    else:
        print("All required packages are already installed.")

if __name__ == "__main__":
    requirements_file = 'requirements.txt'
    compare_requirements_with_installed(requirements_file)
    install_missing_packages(requirements_file)