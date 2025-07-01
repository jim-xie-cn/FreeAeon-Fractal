import cv2
import numpy as np
import matplotlib.pyplot as plt

class CFAImageFourier(object):
    """
    CFAImageFourier provides tools for Fourier analysis of grayscale or RGB images.
    
    Features:
    - Compute raw magnitude and phase components of an image
    - Generate enhanced visualizations for magnitude and phase
    - Reconstruct image from frequency domain
    - Display original, magnitude, phase, and reconstructed images side-by-side
    """

    def __init__(self, image):
        """
        Initialize the object with an input image and compute its Fourier components.
        
        Args:
            image (ndarray): Grayscale or RGB image
        """
        self.m_image = image
        self.m_magnitude = []
        self.m_phase = []
        self.__parse()

    @staticmethod
    def get_image_components(image):
        """
        Compute magnitude and phase from a single image channel using 2D FFT.
        
        Args:
            image (ndarray): Single-channel image
        
        Returns:
            tuple: (magnitude, phase)
        """
        fft_image = np.fft.fft2(image)
        fft_shifted = np.fft.fftshift(fft_image)
        magnitude = np.abs(fft_shifted)
        phase = np.angle(fft_shifted)
        return magnitude, phase

    @staticmethod
    def normalize_and_enhance(array, alpha=1.0, beta=0):
        """
        Normalize an array to 0–255 and apply linear enhancement.
        
        Args:
            array (ndarray): Input data
            alpha (float): Contrast scaling factor
            beta (float): Brightness offset
        
        Returns:
            ndarray: 8-bit image for visualization
        """
        array = cv2.normalize(array, None, 0, 255, cv2.NORM_MINMAX)
        array = np.uint8(array)
        array = cv2.convertScaleAbs(array, alpha=alpha, beta=beta)
        return array

    def __parse(self):
        """
        Internal method: Decompose the image into magnitude and phase,
        handling both grayscale and RGB cases.
        """
        self.m_magnitude = []
        self.m_phase = []

        if self.m_image.ndim == 2 or (self.m_image.ndim == 3 and self.m_image.shape[2] == 1):
            magnitude, phase = CFAImageFourier.get_image_components(self.m_image)
            self.m_magnitude = [magnitude]
            self.m_phase = [phase]

        elif self.m_image.ndim == 3 and self.m_image.shape[2] == 3:
            for c in range(3):
                mag, phs = CFAImageFourier.get_image_components(self.m_image[:, :, c])
                self.m_magnitude.append(mag)
                self.m_phase.append(phs)
        else:
            raise ValueError("Unsupported image format.")

    def get_raw_spectrum(self):
        """
        Get the raw magnitude and phase data (for reconstruction).
        
        Returns:
            tuple: (magnitude list, phase list)
        """
        return self.m_magnitude, self.m_phase

    def get_display_spectrum(self, alpha=1.0, beta=0):
        """
        Generate enhanced visualizations of magnitude and phase for display.
        
        Args:
            alpha (float): Contrast enhancement factor
            beta (float): Brightness offset
        
        Returns:
            tuple: (magnitude images, phase images)
        """
        display_mag = []
        display_phase = []
        for mag_raw, phase_raw in zip(self.m_magnitude, self.m_phase):
            mag_disp = CFAImageFourier.normalize_and_enhance(np.log(1 + mag_raw), alpha=alpha, beta=beta)
            phase_norm = (phase_raw + np.pi) / (2 * np.pi)  # Normalize phase to [0,1]
            phase_disp = CFAImageFourier.normalize_and_enhance(phase_norm, alpha=alpha, beta=beta)
            display_mag.append(mag_disp)
            display_phase.append(phase_disp)
        return display_mag, display_phase

    def get_reconstruct(self):
        """
        Reconstruct the spatial-domain image from stored magnitude and phase.
        
        Returns:
            ndarray: Reconstructed image
        """
        reconstructed_channels = []
        for mag, phase in zip(self.m_magnitude, self.m_phase):
            complex_spectrum = mag * np.exp(1j * phase)
            fft_unshifted = np.fft.ifftshift(complex_spectrum)
            img_reconstructed = np.fft.ifft2(fft_unshifted)
            img_reconstructed = np.real(img_reconstructed)
            img_reconstructed = cv2.normalize(img_reconstructed, None, 0, 255, cv2.NORM_MINMAX)
            reconstructed_channels.append(np.uint8(img_reconstructed))

        if len(reconstructed_channels) == 1:
            return reconstructed_channels[0]
        else:
            return cv2.merge(reconstructed_channels)
            
    def extract_by_freq(self, box):
        """
        Extract a rectangular region from the frequency domain and reconstruct the corresponding spatial image.
    
        Args:
            box (tuple): (x1, y1, x2, y2) coordinates in the frequency domain
    
        Returns:
            ndarray: Reconstructed image from selected frequency region
        """
        x1, y1, x2, y2 = box
        reconstructed_channels = []
    
        for mag, phase in zip(self.m_magnitude, self.m_phase):
            h, w = mag.shape
    
            # Clamp box coordinates within image bounds
            x1_clamped = max(0, min(w, x1))
            x2_clamped = max(0, min(w, x2))
            y1_clamped = max(0, min(h, y1))
            y2_clamped = max(0, min(h, y2))
    
            # Create mask
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[y1_clamped:y2_clamped, x1_clamped:x2_clamped] = 1
    
            # Apply mask and reconstruct
            complex_spectrum = mag * np.exp(1j * phase)
            filtered_spectrum = complex_spectrum * mask
    
            fft_unshifted = np.fft.ifftshift(filtered_spectrum)
            img_reconstructed = np.fft.ifft2(fft_unshifted)
            img_reconstructed = np.real(img_reconstructed)
    
            img_reconstructed = cv2.normalize(img_reconstructed, None, 0, 255, cv2.NORM_MINMAX)
            reconstructed_channels.append(np.uint8(img_reconstructed))
    
        if len(reconstructed_channels) == 1:
            return reconstructed_channels[0]
        else:
            return cv2.merge(reconstructed_channels)
            
    def extract_by_phase(self, box):
        """
        Extract region from phase spectrum only (assuming unit magnitude elsewhere).
        Works for both grayscale and RGB images.
        
        Args:
            box (tuple): (x1, y1, x2, y2) region in frequency domain
    
        Returns:
            ndarray: Reconstructed image from masked phase
        """
        x1, y1, x2, y2 = box
        reconstructed_channels = []
    
        for phase in self.m_phase:
            h, w = phase.shape
    
            # Clamp box boundaries
            x1_clamped = max(0, min(w, x1))
            x2_clamped = max(0, min(w, x2))
            y1_clamped = max(0, min(h, y1))
            y2_clamped = max(0, min(h, y2))
    
            # Default: unit magnitude + zero phase (neutral complex)
            complex_spectrum = np.ones((h, w), dtype=np.complex128)
    
            # Inject real phase in selected region
            complex_spectrum[y1_clamped:y2_clamped, x1_clamped:x2_clamped] = np.exp(
                1j * phase[y1_clamped:y2_clamped, x1_clamped:x2_clamped])
    
            # Inverse FFT
            fft_unshifted = np.fft.ifftshift(complex_spectrum)
            img_reconstructed = np.fft.ifft2(fft_unshifted)
            img_reconstructed = np.real(img_reconstructed)
    
            # Normalize
            img_reconstructed = cv2.normalize(img_reconstructed, None, 0, 255, cv2.NORM_MINMAX)
            reconstructed_channels.append(np.uint8(img_reconstructed))
    
        # Merge for RGB or return grayscale
        if len(reconstructed_channels) == 1:
            return reconstructed_channels[0]
        else:
            return cv2.merge(reconstructed_channels)
        
    def extract_by_freq_phase(self, mag_box, phase_box):
        """
        Extract image region by masking magnitude and phase spectra separately,
        then reconstruct spatial image.
    
        Args:
            mag_box (tuple): (x1, y1, x2, y2) region in magnitude spectrum
            phase_box (tuple): (x1, y1, x2, y2) region in phase spectrum
    
        Returns:
            ndarray: Reconstructed spatial-domain image from masked magnitude and phase regions
        """
        mag_x1, mag_y1, mag_x2, mag_y2 = mag_box
        phase_x1, phase_y1, phase_x2, phase_y2 = phase_box
    
        reconstructed_channels = []
        
        for mag, phase in zip(self.m_magnitude, self.m_phase):
            h, w = mag.shape
    
            mag_mask = np.zeros((h, w), dtype=np.uint8)
            mag_mask[mag_y1:mag_y2, mag_x1:mag_x2] = 1
    
            phase_mask = np.zeros((h, w), dtype=np.uint8)
            phase_mask[phase_y1:phase_y2, phase_x1:phase_x2] = 1
    
            mag_masked = mag * mag_mask
            phase_masked = phase * phase_mask
    
            complex_spectrum = mag_masked * np.exp(1j * phase_masked)
    
            fft_unshifted = np.fft.ifftshift(complex_spectrum)
            img_reconstructed = np.fft.ifft2(fft_unshifted)
            img_reconstructed = np.real(img_reconstructed)
    
            img_reconstructed = cv2.normalize(img_reconstructed, None, 0, 255, cv2.NORM_MINMAX)
            reconstructed_channels.append(np.uint8(img_reconstructed))
    
        if len(reconstructed_channels) == 1:
            return reconstructed_channels[0]
        else:
            return cv2.merge(reconstructed_channels)
        
    def extract_by_freq_mask(self, mask):
        """
        Use a custom binary mask (same size as frequency map) to select which frequency components to retain.
        
        Args:
            mask (ndarray): Binary mask (1: keep, 0: remove), same shape as frequency maps
        
        Returns:
            ndarray: Reconstructed image using the masked frequency domain
        """
        reconstructed_channels = []
        for mag, phase in zip(self.m_magnitude, self.m_phase):
            assert mask.shape == mag.shape, "Mask shape mismatch."
            complex_spectrum = mag * np.exp(1j * phase)
            filtered_spectrum = complex_spectrum * mask
            fft_unshifted = np.fft.ifftshift(filtered_spectrum)
            img_reconstructed = np.fft.ifft2(fft_unshifted)
            img_reconstructed = np.real(img_reconstructed)
            img_reconstructed = cv2.normalize(img_reconstructed, None, 0, 255, cv2.NORM_MINMAX)
            reconstructed_channels.append(np.uint8(img_reconstructed))
        if len(reconstructed_channels) == 1:
            return reconstructed_channels[0]
        else:
            return cv2.merge(reconstructed_channels)

    def plot(self,
             magnitude=[], 
             phase=[], 
             reconstructed=np.array([]),
             region_by_freq=np.array([]), 
             region_by_phase=np.array([]),
             region_by_freq_phase=np.array([]),
             reconstructed_masked=np.array([])):
        """
        Display original, magnitude, phase, reconstructed images,
        and optionally regions extracted from frequency and phase.
        
        Args:
            magnitude (list): List of visualized magnitude images
            phase (list): List of visualized phase images
            reconstructed (ndarray): Reconstructed spatial-domain image
            region_by_freq (ndarray): Extracted region from magnitude spectrum
            region_by_phase (ndarray): Extracted region from phase spectrum
            region_by_freq_phase (ndarray): Extracted region from freq and phase spectrum
            reconstructed_masked (ndarray): Reconstructed by frequency mask
        """
        def enhance_contrast(image, beta=0, min_scale=1, max_scale=1.8):
            std = np.std(image)
            max_std = 160.0  
            scale = max_scale - (std / max_std) * (max_scale - min_scale)
            scale = np.clip(scale, min_scale, max_scale)
            enhanced = cv2.convertScaleAbs(image, alpha=scale, beta=beta)
            return enhanced
        
        def enhance_contrast(image, beta=0, min_scale=1, max_scale=1.8):
            std = np.std(image)
            max_std = 160.0  
            scale = max_scale - (std / max_std) * (max_scale - min_scale)
            scale = np.clip(scale, min_scale, max_scale)
            enhanced = cv2.convertScaleAbs(image, alpha=scale, beta=beta)
            return enhanced

        # Collect all images to display
        images = [
            ("Original", self.m_image),
            ("Magnitude", cv2.merge(magnitude) if len(magnitude) > 1 else magnitude[0] if magnitude else None),
            ("Phase", cv2.merge(phase) if len(phase) > 1 else phase[0] if phase else None),
            ("Reconstructed", reconstructed),
        ]

        if region_by_freq.size != 0:
            images.append(("Region from Freq", region_by_freq))
        if region_by_phase.size != 0:
            images.append(("Region from Phase", region_by_phase))
        if region_by_freq_phase.size != 0:
            images.append(("Region from Freq & Phase", region_by_freq_phase))
        if reconstructed_masked.size != 0:
            images.append(("Reconstructed (Freq Mask)", reconstructed_masked))

        n_total = len(images)
        n_cols = (n_total + 1) // 2
        #plt.figure(figsize=(4 * n_cols, 6 * 2))  # 2 rows
        plt.figure(figsize=(14, 8))

        for idx, (title, img) in enumerate(images):
            plt.subplot(2, n_cols, idx + 1)
            if img is None:
                plt.axis('off')
                continue
            if img.ndim == 2:
                plt.imshow(enhance_contrast(img), cmap='gray', vmin=0, vmax=255)
            else:
                plt.imshow(cv2.cvtColor(enhance_contrast(img), cv2.COLOR_BGR2RGB), vmin=0, vmax=255)
            plt.title(title)
            plt.axis('off')

        plt.tight_layout()
        plt.show()

def main():

    # Read image
    #image = cv2.imread("../images/face.png",cv2.IMREAD_GRAYSCALE)
    image = cv2.imread("../images/face.png")

    # Create CFAImageFourier instance
    fourier = CFAImageFourier(image)

    # Get raw spectrum
    mag_raw, phase_raw = fourier.get_raw_spectrum()

    # Get display spectrum
    mag_disp, phase_disp = fourier.get_display_spectrum(alpha=1.5)

    # Reconstruct full image
    reconstructed = fourier.get_reconstruct()

    # Reconstructet image by frequency mask (reserve odd frequencies))
    h, w = mag_raw[0].shape
    Y, X = np.ogrid[:h, :w]
    mask = ((X % 2 == 1) & (Y % 2 == 1)).astype(np.uint8)
    reconstructed_masked = fourier.extract_by_freq_mask(mask)

    # Get ROI by frequency or phase
    h, w = image.shape[0], image.shape[1]
    freq_box = (0,0,w//2,h//2)
    phase_box = (0,0,w,h)

    region_mag = fourier.extract_by_freq(box=freq_box)
    region_phase = fourier.extract_by_phase(box=phase_box)
    region_mag_phase = fourier.extract_by_freq_phase(freq_box,phase_box)

    # Show full result
    fourier.plot(mag_disp, phase_disp, reconstructed, region_mag, region_phase,region_mag_phase,reconstructed_masked)

if __name__ == "__main__":
    main()
