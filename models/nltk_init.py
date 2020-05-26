import os

from nltk.downloader import Downloader
from nltk import data


class InitNLTK(Downloader):
    '''Check and download necessary nltk resources

    Attributes:
        resource_map_dict, Dict with resource name and it's path
            like key value pairs.
    '''
    def __init__(self, resource_map_dict, **kws):
        super().__init__(**kws)
        self.resource_map = resource_map_dict

    def _download_resource(self, name, path):
        try:
            path = os.path.join(self._get_download_dir(), path)
            data.find(path)
        except LookupError:
            self.download(name)

    def download_resources(self):
        for name, path in self.resource_map.items():
            self._download_resource(name, path)
