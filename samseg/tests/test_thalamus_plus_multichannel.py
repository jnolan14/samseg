import shlex

import numpy as np
import pytest
import surfa as sf

from samseg.subregions import core_plus
from samseg.subregions.core_plus import MeshModelPlus
from samseg.subregions.thalamus_plus import ThalamicNucleiPlus


def _volume(value, shape, voxsize=(1.0, 1.0, 1.0)):
    geometry = sf.ImageGeometry(
        shape, voxsize=voxsize, center=(0.0, 0.0, 0.0))
    return sf.Volume(
        np.full(shape, value, dtype='float32'), geometry=geometry)


class _MRIConvertRunner:

    def __init__(self, mutateOutput=None):
        self.commands = []
        self.mutateOutput = mutateOutput

    def __call__(self, command):
        arguments = shlex.split(command)
        assert arguments[0] == 'mri_convert'
        assert arguments[arguments.index('-odt') + 1] == 'float'
        assert arguments[arguments.index('-rt') + 1] == 'cubic'

        source = sf.load_volume(arguments[1])
        reference = sf.load_volume(
            arguments[arguments.index('-rl') + 1])
        output = reference.new(np.full(
            reference.shape,
            source.data.flat[0],
            dtype='float32'))
        if self.mutateOutput is not None:
            output = self.mutateOutput(output)
        output.save(arguments[2])
        self.commands.append(arguments)


def _configured_model(tmp_path, sourceVolumes):
    model = object.__new__(ThalamicNucleiPlus)
    model.tempDir = str(tmp_path)
    model.resolution = 1.0
    model.synthImage = _volume(10, (6, 6, 6))
    model.inputSeg = _volume(49, (6, 6, 6))
    model.initializationSegmentation = _volume(
        8101, (6, 6, 6))
    model.initializationMask = _volume(1, (6, 6, 6))

    model.inputImageFileNames = []
    model.inputImages = []
    model.correctedImages = []
    for channelNumber, volume in enumerate(sourceVolumes):
        sourceFileName = tmp_path / f'source channel {channelNumber}.mgz'
        volume.save(sourceFileName)
        model.inputImageFileNames.append(str(sourceFileName))
        model.inputImages.append(volume.copy())
        model.correctedImages.append(volume.copy())

    model.intensityPriorReferenceImage = model.inputImages[0]
    mask = np.ones(model.synthImage.shape, dtype='bool')
    mask[1, :, :] = False
    imageMask = model.synthImage.new(mask)
    imageCropping = (slice(1, 5), slice(1, 5), slice(1, 5))
    return model, imageCropping, imageMask


def _assert_same_geometry(first, second):
    assert sf.transform.image_geometry_equal(first, second, tol=1e-5)


def _command_sources(commands):
    return [command[1] for command in commands]


def _command_references(commands):
    return [command[command.index('-rl') + 1] for command in commands]


def test_intensity_channel_resampling_is_mesh_model_plus_machinery():
    assert '_resample_and_stack_intensity_channels' in (
        MeshModelPlus.__dict__)
    assert '_resample_and_stack_intensity_channels' not in (
        ThalamicNucleiPlus.__dict__)


def test_prior_and_regional_stacks_use_independent_reference_geometries(
        tmp_path, monkeypatch):
    model, imageCropping, imageMask = _configured_model(
        tmp_path,
        [
            _volume(11, (6, 6, 6)),
            _volume(22, (12, 12, 12), voxsize=(0.5, 0.5, 0.5)),
        ])
    runner = _MRIConvertRunner()
    monkeypatch.setattr(core_plus.utils, 'run', runner)
    model.resolution = 0.5

    priorImage = model._resample_and_stack_intensity_channels(
        model.inputImageFileNames,
        model.intensityPriorReferenceImage,
        'prior')
    regionalReference = model.synthImage[imageCropping].resize(
        model.resolution, method='nearest')
    regionalMask = imageMask.resample_like(
        regionalReference, method='nearest')
    processedImage = model._resample_and_stack_intensity_channels(
        model.inputImageFileNames,
        regionalReference,
        'regional',
        mask=regionalMask)

    assert priorImage.nframes == 2
    assert processedImage.nframes == 2
    _assert_same_geometry(priorImage, model.intensityPriorReferenceImage)
    _assert_same_geometry(processedImage, regionalReference)
    assert tuple(priorImage.shape[:3]) == (6, 6, 6)
    assert tuple(processedImage.shape[:3]) == (8, 8, 8)

    priorCommands = runner.commands[:2]
    regionalCommands = runner.commands[2:]
    assert _command_sources(priorCommands) == model.inputImageFileNames
    assert _command_sources(regionalCommands) == model.inputImageFileNames
    assert not set(_command_sources(regionalCommands)) & {
        command[2] for command in priorCommands
    }
    assert len(set(_command_references(priorCommands))) == 1
    assert len(set(_command_references(regionalCommands))) == 1
    assert _command_references(priorCommands)[0] != (
        _command_references(regionalCommands)[0])
    assert all('-vs' not in command for command in runner.commands)

    priorData = priorImage.framed_data
    np.testing.assert_array_equal(priorData[..., 0], 11)
    np.testing.assert_array_equal(priorData[..., 1], 22)
    regionalData = processedImage.framed_data
    mask = regionalMask.data != 0
    np.testing.assert_array_equal(regionalData[..., 0][mask], 11)
    np.testing.assert_array_equal(regionalData[..., 1][mask], 22)
    np.testing.assert_array_equal(regionalData[~mask], 0)


def test_resampling_preserves_source_and_successor_state(
        tmp_path, monkeypatch):
    model, imageCropping, imageMask = _configured_model(
        tmp_path, [_volume(11, (6, 6, 6))])
    monkeypatch.setattr(core_plus.utils, 'run', _MRIConvertRunner())
    originalState = {
        'inputImage': model.inputImages[0].data.copy(),
        'correctedImage': model.correctedImages[0].data.copy(),
        'inputSeg': model.inputSeg.data.copy(),
        'synthImage': model.synthImage.data.copy(),
        'initializationSegmentation': (
            model.initializationSegmentation.data.copy()),
        'initializationMask': model.initializationMask.data.copy(),
    }

    model._resample_and_stack_intensity_channels(
        model.inputImageFileNames,
        model.intensityPriorReferenceImage,
        'prior')
    regionalReference = model.synthImage[imageCropping].resize(
        model.resolution, method='nearest')
    regionalMask = imageMask.resample_like(
        regionalReference, method='nearest')
    model._resample_and_stack_intensity_channels(
        model.inputImageFileNames,
        regionalReference,
        'regional',
        mask=regionalMask)

    np.testing.assert_array_equal(
        model.inputImages[0].data, originalState['inputImage'])
    np.testing.assert_array_equal(
        model.correctedImages[0].data, originalState['correctedImage'])
    np.testing.assert_array_equal(
        model.inputSeg.data, originalState['inputSeg'])
    np.testing.assert_array_equal(
        model.synthImage.data, originalState['synthImage'])
    np.testing.assert_array_equal(
        model.initializationSegmentation.data,
        originalState['initializationSegmentation'])
    np.testing.assert_array_equal(
        model.initializationMask.data,
        originalState['initializationMask'])


def test_thalamus_preprocessing_requests_full_and_regional_representations(
        tmp_path, monkeypatch):
    model, _, _ = _configured_model(
        tmp_path,
        [
            _volume(11, (6, 6, 6)),
            _volume(22, (12, 12, 12), voxsize=(0.5, 0.5, 0.5)),
        ])
    model.preliminaryModelProfiles = {'aseg': {}}
    model.inputSegmentationSchemaOverride = None
    model.initializationSegmentation = None
    model.initializationMask = None
    monkeypatch.setattr(
        model, '_configure_preliminary_model_profile',
        lambda *args, **kwargs: None)
    monkeypatch.setattr(
        model, '_ensure_preliminary_model_state', lambda: None)
    monkeypatch.setattr(
        model, '_get_preliminary_affine_support_labels', lambda: [10, 49])
    monkeypatch.setattr(
        model, '_build_preliminary_synthetic_image',
        lambda inputSeg: inputSeg.copy())
    requests = []

    def record_request(sourceFileNames, referenceImage, outputPrefix,
                       mask=None):
        requests.append(
            (list(sourceFileNames), referenceImage, outputPrefix, mask))
        data = np.stack([
            np.full(referenceImage.shape, channelNumber + 1,
                    dtype='float32')
            for channelNumber in range(len(sourceFileNames))
        ], axis=-1)
        if mask is not None:
            data[mask.data == 0] = 0
        return referenceImage.new(data)

    monkeypatch.setattr(
        model, '_resample_and_stack_intensity_channels', record_request)

    model.preprocess_images()

    assert len(requests) == 2
    priorRequest, regionalRequest = requests
    assert priorRequest[0] == model.inputImageFileNames
    assert priorRequest[1] is model.intensityPriorReferenceImage
    assert priorRequest[2] == 'intensityPrior'
    assert priorRequest[3] is None
    assert regionalRequest[0] == model.inputImageFileNames
    assert regionalRequest[2] == 'regionalIntensity'
    assert regionalRequest[3] is model.longMask
    _assert_same_geometry(model.intensityPriorImage, priorRequest[1])
    _assert_same_geometry(model.processedImage, regionalRequest[1])
    assert model.initializationSegmentation is None
    assert model.initializationMask is None


def test_reference_values_do_not_affect_resampled_channel_values(
        tmp_path, monkeypatch):
    model, _, _ = _configured_model(
        tmp_path, [_volume(11, (6, 6, 6))])
    monkeypatch.setattr(core_plus.utils, 'run', _MRIConvertRunner())
    firstReference = _volume(1, (4, 4, 4))
    secondReference = _volume(999, (4, 4, 4))

    first = model._resample_and_stack_intensity_channels(
        model.inputImageFileNames, firstReference, 'first')
    second = model._resample_and_stack_intensity_channels(
        model.inputImageFileNames, secondReference, 'second')

    np.testing.assert_array_equal(first.data, second.data)


def test_mask_must_already_match_requested_reference(tmp_path):
    model, _, _ = _configured_model(
        tmp_path, [_volume(11, (6, 6, 6))])
    reference = _volume(0, (4, 4, 4))
    wrongMask = _volume(1, (5, 5, 5))

    with pytest.raises(ValueError, match='mask must match'):
        model._resample_and_stack_intensity_channels(
            model.inputImageFileNames,
            reference,
            'regional',
            mask=wrongMask)


@pytest.mark.parametrize('invalidOutput', ['frames', 'geometry'])
def test_invalid_resampled_channel_fails_before_stacking(
        tmp_path, monkeypatch, invalidOutput):
    model, _, _ = _configured_model(
        tmp_path, [_volume(11, (6, 6, 6))])

    def invalidate(output):
        if invalidOutput == 'frames':
            return sf.Volume(
                np.stack([output.data, output.data], axis=-1),
                geometry=output.geom)
        matrix = output.geom.vox2world.matrix.copy()
        matrix[0, 3] += 1
        output.geom.vox2world = matrix
        return output

    monkeypatch.setattr(
        core_plus.utils, 'run', _MRIConvertRunner(invalidate))

    with pytest.raises(RuntimeError) as error:
        model._resample_and_stack_intensity_channels(
            model.inputImageFileNames,
            model.intensityPriorReferenceImage,
            'prior')

    assert 'Intensity channel 0' in str(error.value)
    assert model.inputImageFileNames[0] in str(error.value)
