# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['C:\\Users\\ADMIN\\Desktop\\code-clg projec\\tool\\akshita.py'],
    pathex=[],
    binaries=[],
    datas=[('C:\\Users\\ADMIN\\Desktop\\code-clg projec\\tool\\rf_model.pkl', '.'), ('C:\\Users\\ADMIN\\Desktop\\code-clg projec\\tool\\tfidf_vectorizer.pkl', '.'), ('C:\\Users\\ADMIN\\Desktop\\code-clg projec\\tool\\browsing_history.csv', '.')],
    hiddenimports=[],
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
    name='akshita',
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
