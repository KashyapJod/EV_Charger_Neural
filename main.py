import os
# Set environment variables before any other imports
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=INFO, 2=WARNING, 3=ERROR
os.environ['TF_ENABLE_DEPRECATION_WARNINGS'] = '0'
os.environ['TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS'] = '1'
os.environ['KMP_WARNINGS'] = '0'  # Suppress OpenMP warnings
os.environ['CTK_DISABLE_GPU_RENDERING'] = '1'  # Use CPU for GUI rendering

# Now import other modules
import warnings
import tensorflow as tf

# Filter warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

def main():
    # Set optimal coverage radius for better efficiency
    os.environ['DEFAULT_COVERAGE_RADIUS'] = '4.0'  # Slightly reduced for better placement
    
    # Import the app only after environment setup
    from gui.app import EVChargerApp
    app = EVChargerApp()
    app.mainloop()

if __name__ == "__main__":
    main()
