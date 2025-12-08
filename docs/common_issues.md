## Issues during installation

### Long paths in windows

During installation on windows one can encounter the following error:

```
ERROR: Could not install packages due to an OSError: [Errno 2] No such file or directory: 'C:\\Users\\af24\\AppData\\Local\\Temp\\pip-install-i3syxwx0\\imgui-bundle_3462a851e1e44f9caa7b8510206f002e\\external\\hello_imgui\\hello_imgui\\hello_imgui_cmake\\android\\apkCMake\\templates\\sdl\\gradle_template\\app\\src\\main\\java\\org\\libsdl\\app\\HIDDeviceBLESteamController.java'
HINT: This error might have occurred since this system does not have Windows Long Path support enabled. You can find information on how to enable this at https://pip.pypa.io/warnings/enable-long-paths
```

The issues can be fixed by entering Registry Editor and modifying _LongPathsEnabled_ variable under _HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem_ from "0" to "1".

Refer to this webpage on the instructions to fix it:
https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=registry#enable-long-paths-in-windows-10-version-1607-and-later
