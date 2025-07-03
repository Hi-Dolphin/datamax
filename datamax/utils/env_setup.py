import importlib.metadata
import os
import subprocess
import sys


class EnvironmentSetup:
    """Responsible for setting up the correct environment,
    including checking GPU support and installing the necessary packages
    """

    def __init__(self, use_gpu: bool = False):
        self._gpu_available = None
        self._setup_completed = False
        self.use_gpu = use_gpu  # Use GPU if True, otherwise use CPU

    def is_gpu_available(self):
        """Check whether the system supports Gpus"""
        if self._gpu_available is None:
            try:
                # Check whether CUDA is available
                subprocess.check_output(["nvcc", "--version"], stderr=subprocess.STDOUT)
                self._gpu_available = True
            except (subprocess.CalledProcessError, FileNotFoundError):
                self._gpu_available = False
        return self._gpu_available

    def is_conda(self):
        """Check whether the current environment is a Conda environment"""
        return os.path.exists(os.path.join(sys.prefix, "conda-meta"))

    def install_package(self, package_name):
        """Select pip or conda or other installation specified package according to the environment"""
        installer = "conda" if self.is_conda() else "pip"
        if installer == "conda":
            print(f"Detected Conda environment. Installing {package_name} with conda.")
            try:
                subprocess.check_call(["pip", "install", package_name])
                print(f"Successfully installed {package_name} with conda.")
            except subprocess.CalledProcessError as e:
                print(f"Failed to install {package_name} with conda: {e}")
        elif installer == "pip":
            print(f"Using pip to install {package_name}.")
            try:
                # Invoke the pip installation package using the Python interpreter
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", package_name]
                )
                print(f"Successfully installed {package_name} with pip.")
            except subprocess.CalledProcessError as e:
                print(f"Failed to install {package_name} with pip: {e}")
        else:
            print(
                "Unable to determine the package manager. Please install the package manually."
            )

    def check_and_install(self):
        """Check and install appropriate packages based on user's choice and GPU availability"""
        if self._setup_completed:
            return

        # Override GPU detection with the use_gpu parameter
        if self.use_gpu:
            pkg_name = "paddlepaddle-gpu" if self.is_gpu_available() else "paddlepaddle"
        else:
            pkg_name = "paddlepaddle"

        try:
            _ = importlib.metadata.version(
                pkg_name.split()[0]
            )  # Check if paddlepaddle is installed
            # print(f"{pkg_name} version {1} is already installed.")
        except importlib.metadata.PackageNotFoundError:
            print(f"{pkg_name} is not installed. Installing now...")
            self.install_package(pkg_name)

        self._setup_completed = True


# Create an instance of EnvironmentSetup with the desired GPU option and call check_and_install when the program initializes
env_setup = EnvironmentSetup()  # Set this flag as needed


def setup_environment(use_gpu: bool = False):
    """Used to set the environment when the program starts"""
    env_setup.use_gpu = use_gpu
    env_setup.check_and_install()
