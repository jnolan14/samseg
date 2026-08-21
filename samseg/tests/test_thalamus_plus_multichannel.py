import shlex

import numpy as np
import pytest
import surfa as sf

from samseg.subregions import core_plus
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
    # Values 10, 49, and 8101 are distinct mutation sentinels here; this fixture
    # does not test their anatomical meanings.
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


def test_prior_and_regional_stacks_use_independent_reference_geometries(
        tmp_path, monkeypatch):
    """Materialize whole-field and regional channels on independent grids.

    Both representations must resample directly from the original channels.
    Values 11/22 expose channel order, while zeros expose regional masking.
    """
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


def test_intensity_resampling_does_not_modify_source_or_initialization_volumes(
        tmp_path, monkeypatch):
    # Resampling materializes new representations; it must not alter source,
    # corrected, localizer, or fitted-initialization volumes.
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


def test_reference_values_do_not_affect_resampled_channel_values(
        tmp_path, monkeypatch):
    model, _, _ = _configured_model(
        tmp_path, [_volume(11, (6, 6, 6))])
    monkeypatch.setattr(core_plus.utils, 'run', _MRIConvertRunner())
    # Values 1 and 999 make accidental use of reference voxel data visible.
    # Only the otherwise identical reference geometry is authoritative.
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
def test_resampled_channel_with_extra_frames_or_wrong_geometry_is_rejected(
        tmp_path, monkeypatch, invalidOutput):
    model, _, _ = _configured_model(
        tmp_path, [_volume(11, (6, 6, 6))])

    # Extra frames and a shifted affine emulate malformed external-resampler
    # output. Every channel must be one 3-D frame on the requested grid.
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
