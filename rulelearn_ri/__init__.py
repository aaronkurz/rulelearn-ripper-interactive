from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rulelearn-ripper-interactive")
except PackageNotFoundError:
    pass
