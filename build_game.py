import PyInstaller.__main__
import os
import shutil
import sys

# --- CONFIGURATION ---
SCRIPT_NAME = "game5.py"   
EXE_NAME = "Aeroplane Game" 
ICON_NAME = "icon.ico"    

# --- SAFETY CHECK: MODELS FOLDER ---
if not os.path.exists("models"):
    print("\n[ERROR] 'models' folder NOT found!")
    print("Please run 'get_models.py' first to download the offline AI files.")
    sys.exit(1)

# --- CLEAN PREVIOUS BUILDS ---
if os.path.exists("build"): 
    print("Removing old build folder...")
    shutil.rmtree("build")

if os.path.exists("dist"): 
    print("Removing old dist folder...")
    shutil.rmtree("dist")

if os.path.exists(f"{EXE_NAME}.spec"): 
    os.remove(f"{EXE_NAME}.spec")

# Check for Icon
if not os.path.exists(ICON_NAME):
    print(f"WARNING: {ICON_NAME} not found! Building with default icon.")
    icon_arg = []
else:
    print(f"Found icon: {ICON_NAME}")
    icon_arg = [f'--icon={ICON_NAME}']

print(f"--- STARTING ONEDIR BUILD FOR {EXE_NAME} ---")

PyInstaller.__main__.run([
    SCRIPT_NAME,
    f'--name={EXE_NAME}',
    '--noconfirm',
    '--onedir',
    '--noconsole',
    '--clean',
    
    # --- ADD ICON ---
    *icon_arg,
    
    # --- INCLUDE ASSETS ---
    # Syntax: 'source_folder;destination_folder'
    '--add-data=pics;pics',       # Images
    '--add-data=models;models',   # <--- NEW: AI BRAIN (OFFLINE FILES)
    
    # --- COLLECT AI LIBRARIES ---
    '--collect-all=mmpose',
    '--collect-all=mmdet',
    '--collect-all=mmengine',
    '--collect-all=mmcv',
    '--collect-all=sklearn', 
    
    # --- HIDDEN IMPORTS ---
    '--hidden-import=torch',
    '--hidden-import=torchvision',
    '--hidden-import=scipy.special',
    '--hidden-import=scipy.spatial.transform._rotation_groups',
    '--hidden-import=sklearn.utils._typedefs',
    '--hidden-import=sklearn.neighbors._partition_nodes',
])

print(f"--- BUILD COMPLETE! ---")
print(f"Check the 'dist' folder for {EXE_NAME}.exe")
