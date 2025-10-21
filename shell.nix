{ pkgs ? import <nixpkgs> {} }:
let
  version = "311";
  python = pkgs."python${pkgs.lib.versions.majorMinor version}";
  qt = pkgs.libsForQt5; # garante consistência Qt + PyQt
in
pkgs.mkShell {
  venvDir = ".venv";
  packages = (with python.pkgs; [
    venvShellHook
    pip
    pyqt5
    opencv4
    python-uinput
    evdev
  ]) ++ (with pkgs; [
    # Toolchain
    gcc gnumake 
    cmake 
    pkg-config 
    extra-cmake-modules
    stdenv.cc.cc.lib
    zlib 
    zlib.dev

    # OpenGL + Mesa
    libglvnd libGLU mesa
    
    # X11 core
    xorg.libX11 xorg.libXext xorg.libXrender xorg.libXtst xorg.libXi
    xorg.libXrandr xorg.libXcursor xorg.libXdamage xorg.libXfixes
    xorg.libXxf86vm xorg.libxcb xorg.libSM xorg.libICE xorg.libxkbfile
    xorg.libXcomposite xorg.libXinerama
    
    # XCB utils
    xorg.xcbutilkeysyms xorg.xcbutilwm xorg.xcbutilimage
    xorg.xcbutilrenderutil xcb-util-cursor

    xdotool
    
    # Qt stack
    qt.qtbase qt.qtwayland qt.qtsvg
    
    # Extra deps
    libxkbcommon dbus fontconfig freetype
    
    # Backend alternativo
    gtk2-x11 gtk2

    ffmpeg

    nodejs
    nodePackages.npm
    
  ]);

  postVenvCreation = ''
    set -e
    echo ">> Configurando ambiente Python com suporte X11..."
    "$venvDir/bin/pip" install --upgrade pip
    "$venvDir/bin/pip" install torch torchvision \
      --index-url https://download.pytorch.org/whl/rocm6.0
    "$venvDir/bin/pip" uninstall -y opencv-python opencv-contrib-python opencv-python-headless || true
    "$venvDir/bin/pip" install opencv-python-headless ultralytics mss
    "$venvDir/bin/pip" install numpy==1.26
    "$venvDir/bin/pip" install python-xlib || true
  '';

  postShellHook = ''
    export QT_QPA_PLATFORM="xcb"
    export QT_PLUGIN_PATH="${qt.qtbase}/lib/qt-${qt.qtbase.version}/plugins"
    export QT_QPA_PLATFORM_PLUGIN_PATH="${qt.qtbase}/lib/qt-${qt.qtbase.version}/plugins/platforms"
    # unset WAYLAND_DISPLAY
    export GDK_BACKEND=x11
    export CLUTTER_BACKEND=x11
    export SDL_VIDEODRIVER=x11
    export OPENCV_GUI_BACKEND=GTK
    export DISPLAY=''${DISPLAY:-:0}
    export HSA_OVERRIDE_GFX_VERSION=10.3.0
    export HSA_ENABLE_SDMA=0
    export ROC_ENABLE_PRE_VEGA=1
    export LIBGL_ALWAYS_SOFTWARE=0
    export __GLX_VENDOR_LIBRARY_NAME=mesa
    echo ">> Ambiente X11 configurado! Display: $DISPLAY"
  '';
}
