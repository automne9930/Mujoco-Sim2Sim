"""键盘指令：evdev 监听按下/释放，按下即动、松手即停。

↑/↓: 前后  ←/→: 左右  Q/E: 转向  R: 复位
不用 WASD，避免和 MuJoCo viewer 的渲染快捷键冲突。
所有按键都挂在 evdev 上，不依赖 viewer 的 key_callback。
"""

import glob
import threading

import numpy as np

try:
    import evdev
except ImportError:
    evdev = None

_AXES = ("vx", "vy", "wz")
_KEYMAP = {} if evdev is None else {
    evdev.ecodes.KEY_UP: ("vx", +1), evdev.ecodes.KEY_DOWN: ("vx", -1),
    evdev.ecodes.KEY_LEFT: ("vy", +1), evdev.ecodes.KEY_RIGHT: ("vy", -1),
    evdev.ecodes.KEY_Q: ("wz", +1), evdev.ecodes.KEY_E: ("wz", -1),
}


def _find_keyboards(path=None):
    """找到所有带字母键的键盘设备。

    不用 evdev.list_devices()：它要求设备可读写，而 a+r 只给读权限。
    InputDevice 打不开 O_RDWR 时会回退到 O_RDONLY，所以直接 glob 即可。
    """
    if path:
        try:
            return [evdev.InputDevice(path)]
        except (PermissionError, OSError):
            return []

    found = []
    for p in sorted(glob.glob("/dev/input/event*")):
        try:
            dev = evdev.InputDevice(p)
            keys = dev.capabilities().get(evdev.ecodes.EV_KEY, [])
        except (PermissionError, OSError):
            continue
        if evdev.ecodes.KEY_W in keys:
            found.append(dev)
        else:
            dev.close()
    return found


class KeyboardCommand:
    def __init__(self, cfg):
        self._speed = np.array([float(cfg.cmd_vx), float(cfg.cmd_vy), float(cfg.cmd_wz)])
        self._held = set()          # 当前按住的 evdev 键码
        self._reset = False
        self._lock = threading.Lock()

        if evdev is None:
            print("[KEYBOARD] 未安装 evdev，指令恒为 0")
            return

        devs = _find_keyboards(cfg.keyboard_device)
        if not devs:
            print("[KEYBOARD] 无可用键盘（权限？），指令恒为 0")
            print("  修复：sudo chmod a+r /dev/input/event*")
            return

        for dev in devs:
            print(f"[KEYBOARD] {dev.path}: {dev.name}")
            threading.Thread(target=self._read, args=(dev,), daemon=True).start()

    def _read(self, dev):
        try:
            for ev in dev.read_loop():
                if ev.type != evdev.ecodes.EV_KEY:
                    continue
                with self._lock:
                    if ev.code == evdev.ecodes.KEY_R and ev.value == 1:
                        self._reset = True
                    elif ev.code in _KEYMAP:
                        if ev.value == 1:
                            self._held.add(ev.code)
                        elif ev.value == 0:
                            self._held.discard(ev.code)
        except OSError as e:
            print(f"[KEYBOARD] 设备读取中断: {e}")

    def command(self):
        with self._lock:
            held = tuple(self._held)

        vec = np.zeros(3)
        for code in held:
            axis, sign = _KEYMAP[code]
            vec[_AXES.index(axis)] += sign
        return np.clip(vec, -1.0, 1.0) * self._speed

    def consume_reset(self):
        with self._lock:
            reset, self._reset = self._reset, False
        return reset
