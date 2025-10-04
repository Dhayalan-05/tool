# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['kalaiselvi.py'],
    pathex=[],
    binaries=[],
    datas=[('browsing_history.csv', '.'), ('rf_model.pkl', '.'), ('tfidf_vectorizer.pkl', '.')],
    hiddenimports=['importlib.resources'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='kalaiselvi',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
