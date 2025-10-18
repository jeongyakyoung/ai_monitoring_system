# -*- mode: python ; coding: utf-8 -*-


block_cipher = None

datas = [('main.ui', '.'), ('model_conf.ui', '.'), ('telegram.ui', '.'),('relay_port.ui', '.'), ('style/*', 'style/'), ('style/cursor/*', 'style/cursor'), ('style/font/*', 'style/font'), ('controller.py', '.'), ('setting.py', '.'), ('yolo11n-pose_safety.pt', '.'), ('ai.ico', '.'),]

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=['scipy.special.cython_special', 'scipy.special._cdflib'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AI Safety Monitoring',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='ai.ico',
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='main',
)
