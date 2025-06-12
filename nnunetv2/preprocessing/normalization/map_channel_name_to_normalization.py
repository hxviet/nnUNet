from typing import Type
import re

from nnunetv2.preprocessing.normalization.default_normalization_schemes import CTNormalization, NoNormalization, \
    ZScoreNormalization, RescaleTo01Normalization, RGBTo01Normalization, ImageNormalization


channel_name_to_normalization_mapping = {
    'ct': CTNormalization,
    'nonorm': NoNormalization,
    'zscore': ZScoreNormalization,
    'rescale_to_0_1': RescaleTo01Normalization,
    'rgb_to_0_1': RGBTo01Normalization
}

# Cache for dynamically created normalization classes
_dynamic_normalization_classes: dict[str, Type[ImageNormalization]] = {}


def get_normalization_scheme(channel_name: str) -> Type[ImageNormalization]:
    """
    If we find the channel_name in channel_name_to_normalization_mapping return the corresponding normalization. If it is
    not found, we check if it matches the pattern 'ct_to_(lower)_(upper)' where (lower) and (upper) are integers,
    If it matches, we create a normalization class that clips the values in the image to the specified range (CT window) and normalizes the image based on the mean and standard deviation of foreground intensity values across all training images.
    Otherwise, we return ZScoreNormalization as the default normalization scheme.
    """

    norm_scheme = channel_name_to_normalization_mapping.get(channel_name.casefold())

    if norm_scheme:
        return norm_scheme
    
    # Check for custom CT window pattern
    match = re.fullmatch(r'ct_to_(-?\d+)_(-?\d+)', channel_name.casefold())
    if match:
        lower = int(match.group(1))
        upper = int(match.group(2))
        if lower >= upper:
            raise ValueError(f"Invalid CT window: lower bound ({lower}) greater than or equal to upper bound ({upper})")
        
        class_name = f"CTWindow{'Neg' if lower < 0 else ''}{abs(lower)}To{'Neg' if upper < 0 else ''}{abs(upper)}Normalization"
        
        if class_name in _dynamic_normalization_classes:
            norm_scheme = _dynamic_normalization_classes[class_name]
        else:
            def create_run_method(lower_bound, upper_bound):
                def run(self, image, seg=None):
                    assert self.intensityproperties is not None, "Custom range CT normalization requires intensity properties"
                    mean_intensity = self.intensityproperties['mean']
                    std_intensity = self.intensityproperties['std']
                    image = image.astype(self.target_dtype)
                    image = image.clip(lower_bound, upper_bound)
                    image -= mean_intensity
                    image /= max(std_intensity, 1e-8)
                    return image
                return run
            attributes = {
                'leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true': False,
                'run': create_run_method(lower, upper)
            }
            norm_scheme = type(class_name, (ImageNormalization,), attributes)
            _dynamic_normalization_classes[class_name] = norm_scheme
    
    else:
        norm_scheme = ZScoreNormalization
    
    # print('Using %s for image normalization' % norm_scheme.__name__)
    return norm_scheme
