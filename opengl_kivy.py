# Depth buffer must be requested before Kivy opens a window
from kivy.config import Config
Config.set('graphics', 'depth', '16')

import ctypes
import numpy as np

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.graphics import Callback, ClearColor, ClearBuffers
from kivy.graphics.opengl import (
    glEnable, glDisable, glDepthFunc, glViewport,
    glGenBuffers, glBindBuffer, glBufferData,
    glVertexAttribPointer, glEnableVertexAttribArray, glDisableVertexAttribArray,
    glDrawElements, glGetUniformLocation, glGetAttribLocation,
    glUniformMatrix4fv, glUniform3f,
    glUseProgram, glCreateShader, glShaderSource, glCompileShader,
    glGetShaderiv, glGetShaderInfoLog,
    glCreateProgram, glAttachShader, glLinkProgram,
    glGetProgramiv, glGetProgramInfoLog, glDeleteShader,
    GL_DEPTH_TEST, GL_LESS,
    GL_ARRAY_BUFFER, GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW,
    GL_FLOAT, GL_FALSE, GL_TRUE, GL_TRIANGLES, GL_UNSIGNED_SHORT,
    GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, GL_COMPILE_STATUS, GL_LINK_STATUS,
)

from rubiks_states import get_matrix, add_to_hash_set
from rubiks_states import solver as bfs_step

# ── Shaders (GLSL ES 1.00 — compatible with OpenGL ES 2.0 and OpenGL 2.1) ────
# Uses attribute/varying syntax and gl_FragColor instead of the desktop
# layout-qualifier / in-out style so the same source runs on both platforms.

VERT_GLSL = """
attribute vec3 aPos;
attribute vec3 aNormal;

uniform mat4 uMVP;
uniform mat4 uModel;
uniform vec3 uColor;

varying vec3 vColor;
varying vec3 vNormal;
varying vec3 vFragPos;

void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    vFragPos    = vec3(uModel * vec4(aPos, 1.0));
    // mat3(mat4) extracts the upper-left 3x3; correct for rotation + uniform scale
    vNormal     = normalize(mat3(uModel) * aNormal);
    vColor      = uColor;
}
"""

FRAG_GLSL = """
#ifdef GL_ES
precision mediump float;
#endif

varying vec3 vColor;
varying vec3 vNormal;
varying vec3 vFragPos;

uniform vec3 uLightPos;
uniform vec3 uViewPos;

void main() {
    vec3 ambient  = 0.35 * vColor;
    vec3 lightDir = normalize(uLightPos - vFragPos);
    float diff    = max(dot(vNormal, lightDir), 0.0);
    vec3 diffuse  = diff * vColor;
    vec3 viewDir  = normalize(uViewPos - vFragPos);
    vec3 reflDir  = reflect(-lightDir, vNormal);
    float spec    = pow(max(dot(viewDir, reflDir), 0.0), 32.0);
    vec3 specular = 0.25 * spec * vec3(1.0, 1.0, 1.0);
    gl_FragColor  = vec4(ambient + diffuse + specular, 1.0);
}
"""

# ── Cube geometry (all 6 faces, positions + normals) ─────────────────────────

CUBE_VERTS = np.array([
    # position           normal
    # Front  (+z)
    -0.5, -0.5,  0.5,   0,  0,  1,
     0.5, -0.5,  0.5,   0,  0,  1,
     0.5,  0.5,  0.5,   0,  0,  1,
    -0.5,  0.5,  0.5,   0,  0,  1,
    # Back   (-z)
    -0.5, -0.5, -0.5,   0,  0, -1,
    -0.5,  0.5, -0.5,   0,  0, -1,
     0.5,  0.5, -0.5,   0,  0, -1,
     0.5, -0.5, -0.5,   0,  0, -1,
    # Top    (+y)
    -0.5,  0.5, -0.5,   0,  1,  0,
    -0.5,  0.5,  0.5,   0,  1,  0,
     0.5,  0.5,  0.5,   0,  1,  0,
     0.5,  0.5, -0.5,   0,  1,  0,
    # Bottom (-y)
    -0.5, -0.5, -0.5,   0, -1,  0,
     0.5, -0.5, -0.5,   0, -1,  0,
     0.5, -0.5,  0.5,   0, -1,  0,
    -0.5, -0.5,  0.5,   0, -1,  0,
    # Right  (+x)
     0.5, -0.5, -0.5,   1,  0,  0,
     0.5,  0.5, -0.5,   1,  0,  0,
     0.5,  0.5,  0.5,   1,  0,  0,
     0.5, -0.5,  0.5,   1,  0,  0,
    # Left   (-x)
    -0.5, -0.5, -0.5,  -1,  0,  0,
    -0.5, -0.5,  0.5,  -1,  0,  0,
    -0.5,  0.5,  0.5,  -1,  0,  0,
    -0.5,  0.5, -0.5,  -1,  0,  0,
], dtype=np.float32)

# uint16: all 24 vertex indices fit in an unsigned short (ES 2.0 safe)
CUBE_IDX = np.array([
     0,  1,  2,   0,  2,  3,
     4,  5,  6,   4,  6,  7,
     8,  9, 10,   8, 10, 11,
    12, 13, 14,  12, 14, 15,
    16, 17, 18,  16, 18, 19,
    20, 21, 22,  20, 22, 23,
], dtype=np.uint16)

COLORS = [
    (0.90, 0.12, 0.12),  # 1  red
    (0.12, 0.78, 0.12),  # 2  green
    (0.15, 0.25, 0.90),  # 3  blue
    (0.90, 0.88, 0.10),  # 4  yellow
    (0.92, 0.50, 0.10),  # 5  orange
    (0.80, 0.10, 0.80),  # 6  magenta
    (0.10, 0.80, 0.80),  # 7  cyan
    (0.95, 0.95, 0.95),  # 8  white
]

# ── Matrix helpers (row-major; passed to GL with transpose=True) ──────────────

def persp(fov_deg, aspect, near, far):
    f = 1.0 / np.tan(np.radians(fov_deg) / 2)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m

def translate(tx, ty, tz):
    m = np.eye(4, dtype=np.float32)
    m[0, 3] = tx; m[1, 3] = ty; m[2, 3] = tz
    return m

def rot_x(deg):
    r = np.radians(deg); c, s = float(np.cos(r)), float(np.sin(r))
    m = np.eye(4, dtype=np.float32)
    m[1, 1] =  c; m[1, 2] = -s
    m[2, 1] =  s; m[2, 2] =  c
    return m

def rot_y(deg):
    r = np.radians(deg); c, s = float(np.cos(r)), float(np.sin(r))
    m = np.eye(4, dtype=np.float32)
    m[0, 0] =  c; m[0, 2] =  s
    m[2, 0] = -s; m[2, 2] =  c
    return m

def uniform_scale(s):
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = m[1, 1] = m[2, 2] = s
    return m

# ── Solver ────────────────────────────────────────────────────────────────────

def collect_states():
    """BFS over all reachable 2×2 cube states; returns list of state matrices."""
    hash_set = set()
    cube_size = 2
    genesis = get_matrix(cube_size)
    genesis_hash, _ = add_to_hash_set(genesis, hash_set)
    all_states = [genesis.copy()]
    frontier = {genesis_hash: genesis}
    while frontier:
        new_frontier = {}
        for matrix in frontier.values():
            new_states = bfs_step(matrix, hash_set, cube_size)
            for h, m in new_states.items():
                all_states.append(m.copy())
            new_frontier.update(new_states)
        frontier = new_frontier
    return all_states

# ── GL rendering widget ───────────────────────────────────────────────────────

class RubiksWidget(Widget):
    SPACING = 1.08   # centre-to-centre distance between adjacent cubies
    CUBIE   = 0.92   # cubie edge length (leaves a small gap at each face)
    CAM_Z   = 6.0    # camera distance along +z

    def __init__(self, states, **kwargs):
        super().__init__(**kwargs)
        self.states    = states
        self.state_idx = 0
        self.x_rot     = 25.0
        self.y_rot     = -40.0
        self._touch    = None   # (x, y) of last touch point
        self._prog     = None
        self._vbo      = None
        self._ebo      = None
        self._initialized = False

        with self.canvas:
            ClearColor(0.13, 0.13, 0.13, 1.0)
            ClearBuffers(clear_color=True, clear_depth=True)
            self._cb = Callback(self._render)

    # ── Kivy canvas callback ──────────────────────────────────────────────────

    def _render(self, instr):
        if not self._initialized:
            self._setup_gl()
            self._initialized = True
        self._draw_scene()
        # Restore Kivy's expected state — Kivy 2D does not use depth testing
        glDisable(GL_DEPTH_TEST)

    # ── One-time GL setup (runs inside the first render callback) ─────────────

    def _setup_gl(self):
        self._prog = self._build_program(VERT_GLSL, FRAG_GLSL)

        # Upload geometry once; reuse every frame
        self._vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, CUBE_VERTS.nbytes, CUBE_VERTS, GL_STATIC_DRAW)

        self._ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, CUBE_IDX.nbytes, CUBE_IDX, GL_STATIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

        # Cache locations so we don't look them up every frame
        self._u_mvp   = glGetUniformLocation(self._prog, 'uMVP')
        self._u_model = glGetUniformLocation(self._prog, 'uModel')
        self._u_color = glGetUniformLocation(self._prog, 'uColor')
        self._u_light = glGetUniformLocation(self._prog, 'uLightPos')
        self._u_view  = glGetUniformLocation(self._prog, 'uViewPos')
        self._a_pos   = glGetAttribLocation(self._prog,  'aPos')
        self._a_norm  = glGetAttribLocation(self._prog,  'aNormal')

    # ── Per-frame draw ────────────────────────────────────────────────────────

    def _draw_scene(self):
        # Viewport covers just this widget's area (Kivy origin is bottom-left)
        x, y = int(self.x), int(self.y)
        w, h = max(int(self.width), 1), max(int(self.height), 1)
        glViewport(x, y, w, h)

        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glUseProgram(self._prog)

        # Bind geometry and describe layout to the shader
        stride = 6 * 4  # 6 floats × 4 bytes each
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        glVertexAttribPointer(self._a_pos,  3, GL_FLOAT, GL_FALSE, stride,
                              ctypes.c_void_p(0))
        glEnableVertexAttribArray(self._a_pos)
        glVertexAttribPointer(self._a_norm, 3, GL_FLOAT, GL_FALSE, stride,
                              ctypes.c_void_p(12))   # normal starts at byte 12
        glEnableVertexAttribArray(self._a_norm)

        # View-independent uniforms
        proj  = persp(45.0, w / h, 0.1, 100.0)
        view  = translate(0.0, 0.0, -self.CAM_Z)
        orbit = rot_x(self.x_rot) @ rot_y(self.y_rot)
        glUniform3f(self._u_light, 5.0, 7.0, 8.0)
        glUniform3f(self._u_view,  0.0, 0.0, self.CAM_Z)

        matrix = self.states[self.state_idx]
        sz   = matrix.shape[0]
        half = (sz - 1) * self.SPACING * 0.5

        for i in range(sz):
            for j in range(sz):
                for k in range(sz):
                    tx = i * self.SPACING - half
                    ty = j * self.SPACING - half
                    tz = k * self.SPACING - half
                    # scale → translate to position → apply orbit rotation
                    model = orbit @ translate(tx, ty, tz) @ uniform_scale(self.CUBIE)
                    mvp   = proj @ view @ model
                    glUniformMatrix4fv(self._u_mvp,   1, GL_TRUE, mvp)
                    glUniformMatrix4fv(self._u_model, 1, GL_TRUE, model)
                    r, g, b = COLORS[matrix[i, j, k] - 1]
                    glUniform3f(self._u_color, r, g, b)
                    glDrawElements(GL_TRIANGLES, len(CUBE_IDX),
                                   GL_UNSIGNED_SHORT, ctypes.c_void_p(0))

        # Clean up — leave attrib arrays disabled for Kivy's own rendering
        glDisableVertexAttribArray(self._a_pos)
        glDisableVertexAttribArray(self._a_norm)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

    # ── Touch input ───────────────────────────────────────────────────────────

    def on_touch_down(self, touch):
        if self.collide_point(touch.x, touch.y):
            touch.grab(self)
            self._touch = (touch.x, touch.y)
            return True

    def on_touch_move(self, touch):
        if touch.grab_current is self and self._touch:
            dx = touch.x - self._touch[0]
            dy = touch.y - self._touch[1]
            self.y_rot += dx * 0.4
            self.x_rot -= dy * 0.4
            self._touch = (touch.x, touch.y)
            self._cb.ask_update()
            return True

    def on_touch_up(self, touch):
        if touch.grab_current is self:
            touch.ungrab(self)
            self._touch = None
            return True

    def redraw(self):
        """Ask for a canvas redraw; called by the app when the state changes."""
        self._cb.ask_update()

    # ── Shader compilation ────────────────────────────────────────────────────

    @staticmethod
    def _build_program(vs_src, fs_src):
        def compile_shader(src, kind):
            s = glCreateShader(kind)
            glShaderSource(s, src)
            glCompileShader(s)
            if not glGetShaderiv(s, GL_COMPILE_STATUS):
                log = glGetShaderInfoLog(s)
                if isinstance(log, bytes):
                    log = log.decode('utf-8', errors='replace')
                raise RuntimeError(f'Shader compile error:\n{log}')
            return s

        vs   = compile_shader(vs_src, GL_VERTEX_SHADER)
        fs   = compile_shader(fs_src, GL_FRAGMENT_SHADER)
        prog = glCreateProgram()
        glAttachShader(prog, vs)
        glAttachShader(prog, fs)
        glLinkProgram(prog)
        if not glGetProgramiv(prog, GL_LINK_STATUS):
            log = glGetProgramInfoLog(prog)
            if isinstance(log, bytes):
                log = log.decode('utf-8', errors='replace')
            raise RuntimeError(f'Program link error:\n{log}')
        glDeleteShader(vs)
        glDeleteShader(fs)
        return prog


# ── App ───────────────────────────────────────────────────────────────────────

class RubiksApp(App):
    def __init__(self, states, **kwargs):
        super().__init__(**kwargs)
        self.states = states

    def build(self):
        root = BoxLayout(orientation='vertical')

        self.gl_widget = RubiksWidget(self.states, size_hint=(1, 1))

        # Control bar at the bottom
        bar = BoxLayout(size_hint=(1, None), height=60)
        btn_prev = Button(text='< Prev', size_hint=(0.3, 1))
        self.lbl  = Label(text=self._label(), size_hint=(0.4, 1))
        btn_next  = Button(text='Next >', size_hint=(0.3, 1))

        btn_prev.bind(on_release=self._prev)
        btn_next.bind(on_release=self._next)

        bar.add_widget(btn_prev)
        bar.add_widget(self.lbl)
        bar.add_widget(btn_next)

        root.add_widget(self.gl_widget)
        root.add_widget(bar)
        return root

    def _label(self):
        return f'{self.gl_widget.state_idx + 1} / {len(self.states)}'

    def _prev(self, *_):
        self.gl_widget.state_idx = (self.gl_widget.state_idx - 1) % len(self.states)
        self.lbl.text = self._label()
        self.gl_widget.redraw()

    def _next(self, *_):
        self.gl_widget.state_idx = (self.gl_widget.state_idx + 1) % len(self.states)
        self.lbl.text = self._label()
        self.gl_widget.redraw()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Running solver...')
    states = collect_states()
    print(f'Found {len(states)} unique states.')
    RubiksApp(states).run()
