from samseg.subregions.core import MeshModel
from samseg.subregions.core_plus import MeshModelPlus
from samseg.subregions.thalamus import ThalamicNuclei
from samseg.subregions.thalamus_plus import ThalamicNucleiPlus


def test_thalamic_nuclei_plus_switches_base_without_constructor_state_change(
        tmp_path, monkeypatch):
    freesurfer_home = tmp_path / 'freesurfer'
    monkeypatch.setenv('FREESURFER_HOME', str(freesurfer_home))
    arguments = {
        'outDir': str(tmp_path / 'output'),
        'inputImageFileNames': [str(tmp_path / 't1.mgz')],
        'inputSegFileName': str(tmp_path / 'segmentation.mgz'),
    }

    legacy = ThalamicNuclei(**arguments)
    successor = ThalamicNucleiPlus(**arguments)

    assert ThalamicNuclei.__bases__ == (MeshModel,)
    assert ThalamicNucleiPlus.__bases__ == (MeshModelPlus,)
    assert not issubclass(MeshModelPlus, MeshModel)
    assert successor.__dict__ == legacy.__dict__
