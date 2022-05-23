from typing import List, Any


class DataSource:

    def __init__(self, indices: List[int] = None):
        """
        Initialize abstract data source. If no indices are selected, 
        use all the available data.

        Parameters
        ----------
        indices : List[int], optional
            indices of data objects to select, by default None
        """           
        self.source = None
        self.indices = indices

    def __getitem__(self, idx: int) -> Any:
        return self.source[self.indices]
    

'''
class Shape:

    def __init__(
        self,
        points: Tensor,
        dists: Tensor = None,
        matching: Tensor = None,
        faces: Tensor = None,
        geodesics: Tensor = None,
        normals: Tensor = None,
        dims: int = 3,
        index: int = -1
    ):
        self.data = points
        self.dims = dims
        self.dists = dists
        self.match_perm = matching
        self.mp_triv = faces
        self.geo_dist = geodesics
        self.normals = normals
        self.id = index

    @classmethod
    def from_dict(cls, obj: Dict[str, Tensor]) -> Shape:
        return Shape(
            obj['surface'], matching=obj['matching'],
            faces=obj['faces'], normals=obj['normals'],
            dists=obj['dists'], index=obj['index']
        )

    def sample_surface(self, N: int, normals: bool = False) -> Tensor:
        idx = random.sample(range(self.data.shape[0]), N)
        sample = self.data[idx, :]
        if normals:
            sample = torch.cat([sample, self.normals[idx, :]], dim=-1)
        return sample

    def sample_distances(self, N: int, threshold: float = None) -> Tensor:
        if threshold is not None:
            space = self.dists[self.dists[:, 3] <= threshold]
        else:
            space = self.dists
        return space[random.sample(range(space.shape[0]), N), :]

'''