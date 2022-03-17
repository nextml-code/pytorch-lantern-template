from pkg_resources import get_distribution, DistributionNotFound

from . import settings, model

try:
    __version__ = get_distribution("{{cookiecutter.repository_name}}").version
except DistributionNotFound:
    pass
