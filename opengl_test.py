import sys
import ctypes
import numpy as np
from PyQt5.QtCore import QPoint, QSize, Qt
from PyQt5.QtWidgets import QApplication, QOpenGLWidget
from PyQt5.QtGui import QSurfaceFormat
import OpenGL.GL as gl

from rubiks_states import get_matrix, add_to_hash_set
from rubiks_states import solver as bfs_step

# ── Shaders ───────────────────────────────────────────────────────────────────

VERT_SRC = """\
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

uniform mat4 uMVP;
uniform mat4 uModel;
uniform vec3 uColor;

out vec3 vColor;
out vec3 vNormal;
out vec3 vFragPos;

void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    vFragPos    = vec3(uModel * vec4(aPos, 1.0));
    vNormal     = normalize(mat3(transpose(inverse(uModel))) * aNormal);
    vColor      = uColor;
}
"""

FRAG_SRC = """\
#version 330 core
in vec3 vColor;
in vec3 vNormal;
in vec3 vFragPos;

out vec4 FragColor;

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
    vec3 specular = 0.25 * spec * vec3(1.0);
    FragColor = vec4(ambient + diffuse + specular, 1.0);
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

CUBE_IDX = np.array([
     0,  1,  2,   0,  2,  3,
     4,  5,  6,   4,  6,  7,
     8,  9, 10,   8, 10, 11,
    12, 13, 14,  12, 14, 15,
    16, 17, 18,  16, 18, 19,
    20, 21, 22,  20, 22, 23,
], dtype=np.uint32)

# One colour per cell value (1–8)
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

# ── Solver integration ────────────────────────────────────────────────────────

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

# ── GL Widget ─────────────────────────────────────────────────────────────────

class RubiksGLWidget(QOpenGLWidget):
    SPACING = 1.08   # centre-to-centre distance between adjacent cubies
    CUBIE   = 0.92   # cubie edge length (leaves a small gap at each face)
    CAM_Z   = 6.0    # camera distance along +z

    def __init__(self, states, parent=None):
        super().__init__(parent)
        self.states    = states
        self.state_idx = 0
        self.xRot      = 25.0
        self.yRot      = -40.0
        self.lastPos   = QPoint()
        self._prog = self._vao = self._vbo = self._ebo = None
        self.setMinimumSize(600, 600)
        self.setFocusPolicy(Qt.StrongFocus)
        self._update_title()

    def sizeHint(self):
        return QSize(600, 600)

    def initializeGL(self):
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        gl.glClearColor(0.13, 0.13, 0.13, 1.0)

        self._prog = self._build_program(VERT_SRC, FRAG_SRC)

        self._vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self._vao)

        self._vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, CUBE_VERTS.nbytes, CUBE_VERTS, gl.GL_STATIC_DRAW)

        self._ebo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, CUBE_IDX.nbytes, CUBE_IDX, gl.GL_STATIC_DRAW)

        stride = 6 * CUBE_VERTS.itemsize
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride,
                                 ctypes.c_void_p(3 * CUBE_VERTS.itemsize))
        gl.glEnableVertexAttribArray(1)

        gl.glBindVertexArray(0)

    def paintGL(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glUseProgram(self._prog)
        gl.glBindVertexArray(self._vao)

        w, h = max(self.width(), 1), max(self.height(), 1)
        proj  = persp(45.0, w / h, 0.1, 100.0)
        view  = translate(0.0, 0.0, -self.CAM_Z)
        orbit = rot_x(self.xRot) @ rot_y(self.yRot)

        self._setv('uLightPos', np.array([5.0, 7.0, 8.0], dtype=np.float32))
        self._setv('uViewPos',  np.array([0.0, 0.0, self.CAM_Z], dtype=np.float32))

        matrix = self.states[self.state_idx]
        sz   = matrix.shape[0]
        half = (sz - 1) * self.SPACING * 0.5

        for i in range(sz):
            for j in range(sz):
                for k in range(sz):
                    tx = i * self.SPACING - half
                    ty = j * self.SPACING - half
                    tz = k * self.SPACING - half
                    # scale → translate to position → orbit the whole cube
                    model = orbit @ translate(tx, ty, tz) @ uniform_scale(self.CUBIE)
                    mvp   = proj @ view @ model
                    self._setm('uMVP',   mvp)
                    self._setm('uModel', model)
                    self._setv('uColor', np.array(COLORS[matrix[i, j, k] - 1], dtype=np.float32))
                    gl.glDrawElements(gl.GL_TRIANGLES, len(CUBE_IDX), gl.GL_UNSIGNED_INT, None)

        gl.glBindVertexArray(0)

    def resizeGL(self, w, h):
        gl.glViewport(0, 0, w, h)

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key_Right, Qt.Key_Up):
            self.state_idx = (self.state_idx + 1) % len(self.states)
        elif key in (Qt.Key_Left, Qt.Key_Down):
            self.state_idx = (self.state_idx - 1) % len(self.states)
        else:
            return
        self._update_title()
        self.update()

    def mousePressEvent(self, event):
        self.lastPos = event.pos()

    def mouseMoveEvent(self, event):
        dx = event.x() - self.lastPos.x()
        dy = event.y() - self.lastPos.y()
        if event.buttons() & Qt.LeftButton:
            self.xRot += dy * 0.4
            self.yRot += dx * 0.4
            self.update()
        self.lastPos = event.pos()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _update_title(self):
        self.setWindowTitle(
            f"Rubik's States  [{self.state_idx + 1} / {len(self.states)}]"
            "   ← → cycle states   drag to rotate"
        )

    def _u(self, name):
        return gl.glGetUniformLocation(self._prog, name)

    def _setm(self, name, m):
        gl.glUniformMatrix4fv(self._u(name), 1, gl.GL_TRUE, m)

    def _setv(self, name, v):
        gl.glUniform3fv(self._u(name), 1, v)

    @staticmethod
    def _build_program(vs_src, fs_src):
        def compile_shader(src, kind):
            s = gl.glCreateShader(kind)
            gl.glShaderSource(s, src)
            gl.glCompileShader(s)
            if not gl.glGetShaderiv(s, gl.GL_COMPILE_STATUS):
                raise RuntimeError(gl.glGetShaderInfoLog(s).decode())
            return s

        vs   = compile_shader(vs_src, gl.GL_VERTEX_SHADER)
        fs   = compile_shader(fs_src, gl.GL_FRAGMENT_SHADER)
        prog = gl.glCreateProgram()
        gl.glAttachShader(prog, vs)
        gl.glAttachShader(prog, fs)
        gl.glLinkProgram(prog)
        if not gl.glGetProgramiv(prog, gl.GL_LINK_STATUS):
            raise RuntimeError(gl.glGetProgramInfoLog(prog).decode())
        gl.glDeleteShader(vs)
        gl.glDeleteShader(fs)
        return prog


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Must be set before QApplication is created
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setSamples(4)
    QSurfaceFormat.setDefaultFormat(fmt)

    print("Running solver...")
    states = collect_states()
    print(f"Found {len(states)} unique states.")

    app = QApplication(sys.argv)
    window = RubiksGLWidget(states)
    window.show()
    sys.exit(app.exec_())
