# -*- mode: python ; coding: utf-8 -*-

import sys
import site
import os

block_cipher = None

# 라즈베리파이의 Python 패키지 경로 찾기
site_packages_path = site.getsitepackages()[0]
ultralytics_path = os.path.join(site_packages_path, 'ultralytics')

# `datas` 리스트 수정 (필요한 파일 추가)
datas = [
    ('main.ui', '.'), 
    ('model_conf.ui', '.'), 
    ('img', 'img'),
    ('telegram.ui', '.'), 
    ('style/*', 'style/'), 
    ('style/cursor/*', 'style/cursor/'), 
    ('style/font/*', 'style/font/'), 
    ('controller.py', '.'), 
    ('setting.py', '.'), 
    ('yolo11n-pose_safety.pt', '.'), 
    ('ai.ico', '.'), 
    (ultralytics_path, 'ultralytics/')  # Ultralytics 패키지 포함
]

a = Analysis(
    ['main.py'],
    pathex=sys.path,  # 시스템 Python 경로 자동 추가
    binaries=[],
    datas=datas,
    hiddenimports=[
        'scipy.special.cython_special',
        'scipy.special._cdflib',
        'cv2',
        'numpy',
        'asyncio'
    ],
    hookspath=[],
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
    name='AI_Safety_Monitoring',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX 압축 비활성화 (라즈베리파이 호환성 문제 방지)
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
    upx=False,  # UPX 압축 비활성화 (권장)
    upx_exclude=[],
    name='main',
)