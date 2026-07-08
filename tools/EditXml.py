import sys
import os
import math
import re
import numpy as np
import cv2
import ctypes
import xml.etree.ElementTree as ET
from ctypes import *

from PyQt5.QtWidgets import (QApplication, QMainWindow, QGraphicsView, QGraphicsScene,
                             QGraphicsRectItem, QGraphicsPolygonItem, QFileDialog, QListWidget, QVBoxLayout,
                             QHBoxLayout, QWidget, QPushButton, QLabel, QDockWidget,
                             QMessageBox, QGraphicsItem, QAction, QToolBar, QSpinBox,
                             QDoubleSpinBox, QMenu, QAbstractItemView, QDialog, QComboBox,
                             QRadioButton, QProgressBar, QGroupBox, QInputDialog)
from PyQt5.QtCore import Qt, QRectF, QPointF, QSettings, QLineF
from PyQt5.QtGui import QPixmap, QPen, QBrush, QColor, QPainter, QPolygonF

import struct

print(f"Python位数: {struct.calcsize('P') * 8} bit")


# -----------------------------------------------------------------------------
# Halcon DLL 接口封装类 (修复 Halcon 异常崩溃版)
# -----------------------------------------------------------------------------
class MatchResult(Structure):
    _fields_ = [
        ("row", c_double),
        ("col", c_double),
        ("angle", c_double),
        ("scale", c_double),
        ("score", c_double)
    ]


class HalconWrapper:
    def __init__(self, dll_name="TemplateMatchHalcon.dll"):
        self.dll = None
        try:
            # ---------------------------------------------------------
            # 核心修复：智能判断 EXE 所在目录
            # ---------------------------------------------------------
            if getattr(sys, 'frozen', False):
                # 如果是打包后的 EXE，sys.executable 是 exe 文件的全路径
                # os.path.dirname(sys.executable) 就是 exe 所在的目录
                base_dir = os.path.dirname(sys.executable)
            else:
                # 如果是脚本运行，使用 __file__
                base_dir = os.path.dirname(os.path.abspath(__file__))
            # ---------------------------------------------------------

            dll_path = os.path.join(base_dir, dll_name)

            # 调试信息，打包后可以通过命令行运行exe看到这个路径对不对
            print(f"当前运行模式: {'Frozen(EXE)' if getattr(sys, 'frozen', False) else 'Script'}")
            print(f"DLL 搜索目录: {base_dir}")
            print(f"目标 DLL 路径: {dll_path}")

            # 添加 DLL 搜索目录 (Python 3.8+)
            if hasattr(os, 'add_dll_directory'):
                os.add_dll_directory(base_dir)

            # 修改 PATH 环境变量，确保 Halcon 的依赖库也能被找到
            os.environ['PATH'] = base_dir + ';' + os.environ['PATH']

            if os.path.exists(dll_path):
                # winmode=0 是为了让 LoadLibrary 遵循标准的 Windows DLL 搜索顺序
                # 这样它才能找到同目录下的 halcon.dll, halconcpp.dll
                self.dll = cdll.LoadLibrary(dll_path)
                print("DLL 加载成功！")
            else:
                print(f"错误: 找不到文件 {dll_path}")
                # 弹窗提示，方便在非命令行模式下看到错误
                try:
                    import ctypes.wintypes
                    ctypes.windll.user32.MessageBoxW(0, f"找不到DLL文件:\n{dll_path}\n请确保所有DLL都在EXE同级目录下。",
                                                     "错误", 0x10)
                except:
                    pass

        except Exception as e:
            print(f"DLL 加载异常: {e}")
            import traceback
            traceback.print_exc()
            try:
                import ctypes.wintypes
                ctypes.windll.user32.MessageBoxW(0, f"DLL加载异常:\n{str(e)}", "错误", 0x10)
            except:
                pass

    def make_template(self, img_np, save_path, start_angle=-180, angle_extent=360, scale_min=0.8, scale_max=1.2):
        if not self.dll: return False

        # 1. 图像有效性检查 (防止 0xC0000409 崩溃的核心)
        # 多边形Mask后可能大部分是黑色，导致Halcon无法提取特征从而抛出未捕获异常
        if img_np is None or img_np.size == 0: return False

        # 计算标准差，如果对比度太低(纯色/纯黑)，Halcon会崩
        mean, stddev = cv2.meanStdDev(img_np)
        if stddev[0][0] < 2.0:
            print("错误: 模版图像对比度过低或为空，Halcon无法制作模版。")
            return False

        height, width = img_np.shape[:2]
        channels = 1 if len(img_np.shape) == 2 else img_np.shape[2]

        # 2. 内存清洗：确保数据绝对连续
        # 多边形操作后的 numpy array 往往不连续
        img_data = np.ascontiguousarray(img_np, dtype=np.uint8)
        p_data = img_data.ctypes.data_as(POINTER(c_ubyte))

        params = (c_float * 20)()
        params[0] = float(start_angle)
        params[1] = float(angle_extent)
        params[2] = float(scale_min)
        params[3] = float(scale_max)

        p_path = create_string_buffer(save_path.encode('gbk'))  # GBK 防止中文路径崩溃

        try:
            func = self.dll.halconTemplateMakeCshap
            func.argtypes = [POINTER(c_ubyte), c_int, c_int, c_int, c_char_p, POINTER(c_float)]
            func.restype = c_bool

            return func(p_data, height, width, channels, p_path, params)
        except Exception as e:
            print(f"Make Template Error: {e}")
            return False

    def match_template(self, img_np, model_path, settings):
        if not self.dll: return []

        # 1. 内存清洗
        height, width = img_np.shape[:2]
        channels = 1 if len(img_np.shape) == 2 else img_np.shape[2]
        img_data = np.ascontiguousarray(img_np, dtype=np.uint8)
        p_data = img_data.ctypes.data_as(POINTER(c_ubyte))

        # 2. 输出缓冲区修复 (必不可少)
        # C++ 代码逻辑：int dstWidth = width; if (width % 4 != 0) dstWidth = width + 4 - width % 4;
        # 并且转成了 BGR (3通道)
        dst_width = width
        if width % 4 != 0:
            dst_width = width + 4 - (width % 4)

        out_size = height * dst_width * 3
        out_buffer = create_string_buffer(out_size)
        p_out = cast(out_buffer, POINTER(c_ubyte))

        p_path = create_string_buffer(model_path.encode('gbk'))

        # 3. 参数与结果数组
        params = (c_float * 20)()
        params[0] = -180.0;
        params[1] = 360.0;
        params[2] = 0.5;
        params[3] = 1.5
        params[4] = float(settings['min_score'])
        params[5] = float(settings['num_matches'])
        params[6] = float(settings['max_overlap'])
        params[7] = 0.0;
        params[8] = 0.0
        params[9] = float(settings['greediness'])

        max_results = 2000  # 足够大防止溢出
        results_arr = (MatchResult * max_results)()
        match_num = c_int(0)

        try:
            func = self.dll.halconTemplateMatchCshap
            func.argtypes = [POINTER(c_ubyte), c_int, c_int, c_int, POINTER(c_ubyte), c_char_p, POINTER(c_float),
                             POINTER(MatchResult), POINTER(c_int)]
            func.restype = c_bool

            success = func(p_data, height, width, channels, p_out, p_path, params, results_arr, byref(match_num))

            final_results = []
            if success:
                count = min(match_num.value, max_results)
                for i in range(count):
                    res = results_arr[i]
                    final_results.append({
                        'row': res.row, 'col': res.col, 'angle': res.angle, 'scale': res.scale, 'score': res.score
                    })
            return final_results
        except Exception as e:
            print(f"Match Error: {e}")
            return []


halcon_api = HalconWrapper()


# -----------------------------------------------------------------------------
# 辅助函数
# -----------------------------------------------------------------------------
def is_rect_overlap(r1, r2):
    if (r1['x'] >= r2['x'] + r2['w']) or (r1['x'] + r1['w'] <= r2['x']) or \
            (r1['y'] >= r2['y'] + r2['h']) or (r1['y'] + r1['h'] <= r2['y']):
        return False
    return True


def get_chip_index(filename):
    basename = os.path.basename(filename)
    match = re.search(r'Chip_(\d+)', basename, re.IGNORECASE)
    if match: return int(match.group(1))
    return None


# -----------------------------------------------------------------------------
# 匹配配置对话框
# -----------------------------------------------------------------------------
class MatchConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Halcon 模板匹配配置")
        self.resize(400, 400)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        gb_scope = QGroupBox("匹配范围")
        layout_scope = QHBoxLayout()
        self.rb_current = QRadioButton("仅当前图片")
        self.rb_all = QRadioButton("列表所有图片")
        self.rb_current.setChecked(True)
        layout_scope.addWidget(self.rb_current);
        layout_scope.addWidget(self.rb_all)
        gb_scope.setLayout(layout_scope);
        layout.addWidget(gb_scope)

        gb_param = QGroupBox("Halcon 参数")
        form_layout = QVBoxLayout()
        h1 = QHBoxLayout();
        h1.addWidget(QLabel("最小分数 (0-1):"));
        self.spin_score = QDoubleSpinBox()
        self.spin_score.setRange(0.1, 1.0);
        self.spin_score.setValue(0.5);
        self.spin_score.setSingleStep(0.05)
        h1.addWidget(self.spin_score);
        form_layout.addLayout(h1)

        h2 = QHBoxLayout();
        h2.addWidget(QLabel("最大个数:"));
        self.spin_num = QSpinBox()
        self.spin_num.setRange(1, 1000);
        self.spin_num.setValue(10)
        h2.addWidget(self.spin_num);
        form_layout.addLayout(h2)

        h3 = QHBoxLayout();
        h3.addWidget(QLabel("最大重叠 (0-1):"));
        self.spin_overlap = QDoubleSpinBox()
        self.spin_overlap.setRange(0.0, 1.0);
        self.spin_overlap.setValue(0.1);
        self.spin_overlap.setSingleStep(0.1)
        h3.addWidget(self.spin_overlap);
        form_layout.addLayout(h3)

        h4 = QHBoxLayout();
        h4.addWidget(QLabel("贪婪度 (0-1):"));
        self.spin_greed = QDoubleSpinBox()
        self.spin_greed.setRange(0.0, 1.0);
        self.spin_greed.setValue(1);
        self.spin_greed.setSingleStep(0.1)
        h4.addWidget(self.spin_greed);
        form_layout.addLayout(h4)

        gb_param.setLayout(form_layout);
        layout.addWidget(gb_param)
        btn_layout = QHBoxLayout();
        self.btn_run = QPushButton("开始匹配")
        self.btn_run.clicked.connect(self.accept);
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_run);
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)

    def get_settings(self):
        return {
            "scope_all": self.rb_all.isChecked(), "min_score": self.spin_score.value(),
            "num_matches": self.spin_num.value(), "max_overlap": self.spin_overlap.value(),
            "greediness": self.spin_greed.value()
        }


# -----------------------------------------------------------------------------
# 自定义图形项
# -----------------------------------------------------------------------------
class ResizableRectItem(QGraphicsRectItem):
    handle_size = 10.0
    handleTopLeft = 1;
    handleTopMiddle = 2;
    handleTopRight = 3
    handleMiddleLeft = 4;
    handleMiddleRight = 5
    handleBottomLeft = 6;
    handleBottomMiddle = 7;
    handleBottomRight = 8

    handleCursors = {
        handleTopLeft: Qt.SizeFDiagCursor, handleTopMiddle: Qt.SizeVerCursor, handleTopRight: Qt.SizeBDiagCursor,
        handleMiddleLeft: Qt.SizeHorCursor, handleMiddleRight: Qt.SizeHorCursor,
        handleBottomLeft: Qt.SizeBDiagCursor, handleBottomMiddle: Qt.SizeVerCursor,
        handleBottomRight: Qt.SizeFDiagCursor,
    }

    def __init__(self, x, y, w, h, roi_para=0, on_change_callback=None, canvas_ref=None):
        super().__init__(0, 0, w, h)
        self.setPos(x, y)
        self.roi_para = roi_para
        self.on_change_callback = on_change_callback
        self.canvas_ref = canvas_ref
        self.setPen(QPen(Qt.green, 2));
        self.setBrush(QBrush(QColor(0, 255, 0, 50)))
        self.setFlags(
            QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)
        self.handleSelected = None;
        self.mousePressPos = None;
        self.mousePressRect = None

    def trigger_update(self):
        if self.on_change_callback:
            scene_pos = self.scenePos();
            rect = self.rect()
            self.on_change_callback(int(scene_pos.x() + rect.x()), int(scene_pos.y() + rect.y()), int(rect.width()),
                                    int(rect.height()))

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange and self.scene():
            new_pos = value;
            rect = self.rect();
            scene_rect = self.scene().sceneRect()
            curr_l = new_pos.x() + rect.left();
            curr_t = new_pos.y() + rect.top()
            curr_r = new_pos.x() + rect.right();
            curr_b = new_pos.y() + rect.bottom()
            dx = 0;
            dy = 0
            if curr_l < 0:
                dx = -curr_l
            elif curr_r > scene_rect.width():
                dx = scene_rect.width() - curr_r
            if curr_t < 0:
                dy = -curr_t
            elif curr_b > scene_rect.height():
                dy = scene_rect.height() - curr_b
            if dx != 0 or dy != 0:
                new_pos.setX(new_pos.x() + dx);
                new_pos.setY(new_pos.y() + dy);
                value = new_pos
            self.trigger_update()
        return super().itemChange(change, value)

    def handleAt(self, point):
        rect = self.rect();
        s = self.handle_size
        if QRectF(rect.left(), rect.top(), s, s).contains(point): return self.handleTopLeft
        if QRectF(rect.right() - s, rect.top(), s, s).contains(point): return self.handleTopRight
        if QRectF(rect.left(), rect.bottom() - s, s, s).contains(point): return self.handleBottomLeft
        if QRectF(rect.right() - s, rect.bottom() - s, s, s).contains(point): return self.handleBottomRight
        return None

    def hoverMoveEvent(self, moveEvent):
        if self.isSelected():
            handle = self.handleAt(moveEvent.pos())
            self.setCursor(self.handleCursors[handle] if handle else Qt.ArrowCursor)
        super().hoverMoveEvent(moveEvent)

    def hoverLeaveEvent(self, moveEvent):
        self.setCursor(Qt.ArrowCursor); super().hoverLeaveEvent(moveEvent)

    def mousePressEvent(self, mouseEvent):
        if mouseEvent.button() == Qt.LeftButton:
            self.handleSelected = self.handleAt(mouseEvent.pos())
            if self.handleSelected:
                self.mousePressPos = mouseEvent.pos();
                self.mousePressRect = self.rect()
        super().mousePressEvent(mouseEvent);
        self.trigger_update()

    def mouseMoveEvent(self, mouseEvent):
        if self.handleSelected is not None:
            self.interactiveResize(mouseEvent.pos()); self.trigger_update()
        else:
            super().mouseMoveEvent(mouseEvent); self.trigger_update() if self.isSelected() else None

    def mouseReleaseEvent(self, mouseEvent):
        super().mouseReleaseEvent(mouseEvent);
        self.handleSelected = None;
        self.update();
        self.trigger_update()

    def contextMenuEvent(self, event):
        menu = QMenu()
        del_action = menu.addAction("删除 ROI")
        menu.addSeparator()
        match_action = menu.addAction("Halcon 模板匹配...")
        action = menu.exec_(event.screenPos())
        if action == del_action:
            if self.scene():
                self.scene().removeItem(self)
                if self.canvas_ref and self.canvas_ref.main_window_ref:
                    self.canvas_ref.main_window_ref.update_roi_status()
        elif action == match_action:
            if self.canvas_ref and self.canvas_ref.main_window_ref:
                self.canvas_ref.main_window_ref.prepare_template_from_item(self)

    def interactiveResize(self, mousePos):
        scene_pos = self.mapToScene(mousePos);
        scene_rect = self.scene().sceneRect()
        cx = min(max(scene_pos.x(), 0), scene_rect.width())
        cy = min(max(scene_pos.y(), 0), scene_rect.height())
        local_pos = self.mapFromScene(QPointF(cx, cy))
        rect = self.mousePressRect;
        diff = local_pos - self.mousePressPos
        self.prepareGeometryChange();
        new_rect = QRectF(rect)
        if self.handleSelected == self.handleTopLeft:
            new_rect.setTopLeft(rect.topLeft() + diff)
        elif self.handleSelected == self.handleTopRight:
            new_rect.setTopRight(rect.topRight() + diff)
        elif self.handleSelected == self.handleBottomLeft:
            new_rect.setBottomLeft(rect.bottomLeft() + diff)
        elif self.handleSelected == self.handleBottomRight:
            new_rect.setBottomRight(rect.bottomRight() + diff)
        self.setRect(new_rect.normalized());
        self.update()


# -----------------------------------------------------------------------------
# 图像画布视图
# -----------------------------------------------------------------------------
class ImageCanvas(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.NoDrag)

        self.mode = 'rect'
        self.drawing = False
        self.start_point = QPointF()
        self.current_roi_item = None
        self.roi_para_counter = 0

        self.poly_points = []
        self.temp_poly_item = None

        self.scene.selectionChanged.connect(self.on_selection_changed)
        self.main_window_ref = None

    def set_image(self, pixmap):
        self.scene.clear()
        # [修复] 场景清空后，所有Item都被销毁，必须重置引用
        self.temp_poly_item = None
        self.current_roi_item = None
        self.current_image_item = self.scene.addPixmap(pixmap)
        self.setSceneRect(QRectF(pixmap.rect()))
        self.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        self.poly_points = []
        if self.main_window_ref: self.main_window_ref.update_roi_status()

    def add_roi(self, x, y, w, h, para):
        rect_item = ResizableRectItem(x, y, w, h, para, self.update_roi_info, self)
        self.scene.addItem(rect_item)
        if self.main_window_ref: self.main_window_ref.update_roi_status()
        return rect_item

    def set_mode(self, mode):
        self.mode = mode
        self.poly_points = []

        # [修复] 安全移除
        if self.temp_poly_item:
            try:
                if self.temp_poly_item.scene() == self.scene:
                    self.scene.removeItem(self.temp_poly_item)
            except RuntimeError:
                pass
            self.temp_poly_item = None

        if mode == 'polygon':
            self.setDragMode(QGraphicsView.NoDrag)
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            zoom_in = event.angleDelta().y() > 0
            scale = 1.25 if zoom_in else 0.8
            self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
            self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
            self.scale(scale, scale)
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event):
        if self.mode == 'polygon':
            if event.button() == Qt.LeftButton:
                pos = self.mapToScene(event.pos())
                rect = self.scene.sceneRect()
                x = min(max(pos.x(), 0), rect.width())
                y = min(max(pos.y(), 0), rect.height())
                self.poly_points.append(QPointF(x, y))
                self.update_poly_display()
                return
            elif event.button() == Qt.RightButton:
                self.finish_polygon()
                return

        item = self.itemAt(event.pos())
        if isinstance(item, ResizableRectItem): super().mousePressEvent(event); return
        if event.button() == Qt.LeftButton and self.mode == 'rect':
            self.drawing = True;
            pos = self.mapToScene(event.pos())
            rect = self.scene.sceneRect()
            x = min(max(pos.x(), 0), rect.width());
            y = min(max(pos.y(), 0), rect.height())
            self.start_point = QPointF(x, y)
            self.current_roi_item = ResizableRectItem(x, y, 0, 0, self.roi_para_counter, self.update_roi_info, self)
            self.scene.addItem(self.current_roi_item)
        else:
            super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.mode == 'polygon' and event.button() == Qt.LeftButton:
            self.finish_polygon()
        else:
            super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event):
        if self.mode == 'polygon': super().mouseMoveEvent(event); return

        if self.drawing and self.current_roi_item and self.mode == 'rect':
            pos = self.mapToScene(event.pos());
            rect = self.scene.sceneRect()
            cx = min(max(pos.x(), 0), rect.width());
            cy = min(max(pos.y(), 0), rect.height())
            tlx = min(self.start_point.x(), cx);
            tly = min(self.start_point.y(), cy)
            w = abs(self.start_point.x() - cx);
            h = abs(self.start_point.y() - cy)
            self.current_roi_item.setRect(0, 0, w, h)
            self.current_roi_item.setPos(tlx, tly)
            self.current_roi_item.trigger_update()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.mode == 'rect' and self.drawing and event.button() == Qt.LeftButton:
            self.drawing = False
            if self.current_roi_item:
                if self.current_roi_item.rect().width() < 5 or self.current_roi_item.rect().height() < 5:
                    self.scene.removeItem(self.current_roi_item)
                else:
                    self.current_roi_item.setSelected(True)
                    self.current_roi_item.trigger_update()
                    if self.main_window_ref: self.main_window_ref.update_roi_status()
            self.current_roi_item = None
        super().mouseReleaseEvent(event)

    def update_poly_display(self):
        # [修复] 安全移除旧的 Item
        if self.temp_poly_item:
            try:
                # 检查该 item 是否属于当前 scene
                if self.temp_poly_item.scene() == self.scene:
                    self.scene.removeItem(self.temp_poly_item)
            except RuntimeError:
                # 如果对象已被C++销毁，捕获异常并忽略
                pass
            self.temp_poly_item = None  # 移除后立即置空

        if len(self.poly_points) > 0:
            poly = QPolygonF(self.poly_points)
            self.temp_poly_item = QGraphicsPolygonItem(poly)
            self.temp_poly_item.setPen(QPen(Qt.red, 2))
            self.temp_poly_item.setBrush(QBrush(QColor(255, 0, 0, 50)))
            self.scene.addItem(self.temp_poly_item)

    def finish_polygon(self):
        if len(self.poly_points) < 3:
            self.poly_points = []
            self.update_poly_display();
            return
        if self.main_window_ref: self.main_window_ref.create_template_from_polygon(self.poly_points)
        self.poly_points = []
        self.update_poly_display()
        self.set_mode('rect')

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            deleted = False
            for item in self.scene.selectedItems():
                if isinstance(item, ResizableRectItem):
                    self.scene.removeItem(item);
                    deleted = True
            if deleted and self.main_window_ref: self.main_window_ref.update_roi_status()
        elif event.key() == Qt.Key_Escape and self.mode == 'polygon':
            self.set_mode('rect')
        super().keyPressEvent(event)

    def on_selection_changed(self):
        items = self.scene.selectedItems()
        if items and isinstance(items[0], ResizableRectItem): items[0].trigger_update()

    def update_roi_info(self, x, y, w, h):
        if self.main_window_ref:
            self.main_window_ref.status_label.setText(
                f"Selection: X={x}, Y={y}, W={w}, H={h} | Total ROIs: {self.main_window_ref.get_roi_count()}")


# -----------------------------------------------------------------------------
# 主窗口
# -----------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ROI 编辑器")
        self.resize(1200, 800)
        self.settings = QSettings("MyCompany", "RoiEditorApp")

        self.all_rois = {}
        self.image_paths = []
        self.current_img_idx = -1
        self.xml_tree = None
        self.template_path = os.path.abspath("temp_template.shm")
        self.template_w = 0
        self.template_h = 0

        self.init_ui()

    def init_ui(self):
        self.status_label = QLabel("Ready")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.statusBar().addWidget(self.status_label, 1)
        self.statusBar().addWidget(self.progress_bar, 0)

        self.canvas = ImageCanvas()
        self.canvas.main_window_ref = self
        self.setCentralWidget(self.canvas)

        dock_w = QWidget();
        dock_l = QVBoxLayout(dock_w)
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list_widget.clicked.connect(self.on_list_clicked)
        self.list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self.show_list_context_menu)
        dock_l.addWidget(self.list_widget)

        btn_l = QHBoxLayout()
        btn_clr = QPushButton("清空");
        btn_del = QPushButton("删除")
        btn_up = QPushButton("上移");
        btn_down = QPushButton("下移")
        btn_clr.clicked.connect(self.clear_images);
        btn_del.clicked.connect(self.delete_current_image)
        btn_up.clicked.connect(self.move_image_up);
        btn_down.clicked.connect(self.move_image_down)
        btn_l.addWidget(btn_clr);
        btn_l.addWidget(btn_del);
        btn_l.addWidget(btn_up);
        btn_l.addWidget(btn_down)
        dock_l.addLayout(btn_l)
        dock = QDockWidget("图片列表", self);
        dock.setWidget(dock_w)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

        toolbar = QToolBar("Tools");
        self.addToolBar(toolbar)
        act_load = QAction("加载图片", self);
        act_load.triggered.connect(self.load_images)
        act_xml_l = QAction("加载XML", self);
        act_xml_l.triggered.connect(self.load_xml)
        act_xml_s = QAction("保存XML", self);
        act_xml_s.triggered.connect(self.save_xml)
        toolbar.addAction(act_load);
        toolbar.addAction(act_xml_l);
        toolbar.addAction(act_xml_s)
        toolbar.addSeparator()

        # 新增多边形模版绘制按钮
        act_poly_draw = QAction("绘制多边形模版", self)
        act_poly_draw.triggered.connect(self.start_poly_draw)
        toolbar.addAction(act_poly_draw)

        toolbar.addSeparator()
        act_copy_to = QAction("复制到...", self);
        act_copy_to.triggered.connect(self.copy_to_any_image)
        toolbar.addAction(act_copy_to)

        act_copy = QAction("复制上一张(无重叠)", self);
        act_copy.triggered.connect(self.copy_previous_roi)
        act_clean = QAction("清空当前ROI", self);
        act_clean.triggered.connect(self.clear_current_rois_on_canvas)
        toolbar.addAction(act_copy);
        toolbar.addAction(act_clean)
        toolbar.addSeparator()

        toolbar.addWidget(QLabel(" Para (Chip ID): "))
        self.spin_para = QSpinBox();
        self.spin_para.setRange(0, 999999)
        self.spin_para.valueChanged.connect(self.on_para_changed)
        toolbar.addWidget(self.spin_para)

    def get_roi_count(self):
        return len([i for i in self.canvas.scene.items() if isinstance(i, ResizableRectItem)])

    def update_roi_status(self, msg="Ready"):
        self.status_label.setText(f"{msg} | 当前图片ROI数量: {self.get_roi_count()}")

    def on_list_clicked(self):
        self.save_current_rois_to_memory()
        self.current_img_idx = self.list_widget.currentRow()
        self.refresh_current_scene()

    def refresh_current_scene(self):
        if self.current_img_idx < 0 or self.current_img_idx >= len(self.image_paths): return
        path = self.image_paths[self.current_img_idx]
        idx_from_name = get_chip_index(path)
        if idx_from_name is not None:
            self.spin_para.setValue(idx_from_name)
        else:
            self.spin_para.setValue(self.all_rois.get(path, [{'para': 0}])[0]['para'] if self.all_rois.get(path) else 0)
        self.canvas.roi_para_counter = self.spin_para.value()
        if os.path.exists(path):
            self.canvas.set_image(QPixmap(path))
            for r in self.all_rois.get(path, []): self.canvas.add_roi(r['x'], r['y'], r['w'], r['h'], r['para'])
        else:
            QMessageBox.warning(self, "Error", f"Image not found: {path}")
        self.update_roi_status()

    # ------------------ 多边形模版制作 (核心修复：有效性检查+深度清洗) ------------------
    def start_poly_draw(self):
        if self.current_img_idx == -1:
            QMessageBox.warning(self, "提示", "请先加载并选中一张图片")
            return
        self.canvas.set_mode('polygon')
        self.status_label.setText("模式: 绘制多边形模版 | 左键加点 | 双击/右键完成")

    def create_template_from_polygon(self, poly_points):
        if len(poly_points) < 3: return
        img_path = self.image_paths[self.current_img_idx]
        img_np = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img_np is None: return

        pts = np.array([(int(p.x()), int(p.y())) for p in poly_points], dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        h_img, w_img = img_np.shape[:2]

        # 边界检查
        x = max(0, x);
        y = max(0, y)
        if x + w > w_img: w = w_img - x
        if y + h > h_img: h = h_img - y
        if w <= 0 or h <= 0: return

        # 裁剪 + Mask
        # 【重要】必须使用 .copy() 断开与原图的内存视图关系
        crop_img = img_np[y:y + h, x:x + w].copy()

        # 制作掩膜
        pts_local = pts - np.array([x, y])
        mask = np.zeros_like(crop_img)
        cv2.fillPoly(mask, [pts_local], 255)

        mask=255-mask
        masked_template=cv2.inpaint(crop_img,mask,31,cv2.INPAINT_NS)
        # 逻辑与，背景变黑
        # masked_template = cv2.bitwise_and(crop_img, crop_img, mask=mask)

        # 再次确保内存连续性 (Numpy在做bitwise运算后可能产生不连续内存)
        masked_template = np.ascontiguousarray(masked_template)
        cv2.imwrite('masked_template.bmp',masked_template)

        self.template_w = w;
        self.template_h = h

        if not halcon_api.dll:
            QMessageBox.critical(self, "Error", "Halcon DLL 未加载")
            return

        success = halcon_api.make_template(masked_template, self.template_path)
        if success:
            reply = QMessageBox.question(self, "成功", "多边形模板已生成，是否立即匹配？",
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                dlg = MatchConfigDialog(self)
                if dlg.exec_() == QDialog.Accepted:
                    self.run_matching_process(dlg.get_settings())
        else:
            QMessageBox.critical(self, "Error", "模板生成失败(可能是对比度不足)")

    # ------------------ 匹配执行逻辑 ------------------
    def run_matching_process(self, settings):
        if not os.path.exists(self.template_path): return
        self.save_current_rois_to_memory()
        scope_all = settings["scope_all"]
        target_indices = range(len(self.image_paths)) if scope_all else [self.current_img_idx]
        if scope_all:
            self.progress_bar.setVisible(True);
            self.progress_bar.setRange(0, len(target_indices));
            self.progress_bar.setValue(0)
        match_total = 0
        for i, idx in enumerate(target_indices):
            path = self.image_paths[idx]
            new_rois = self.match_single_image(path, settings)
            if new_rois:
                self.all_rois[path] = self.all_rois.get(path, []) + new_rois
                match_total += len(new_rois)
            if scope_all: self.progress_bar.setValue(i + 1); QApplication.processEvents()
        self.progress_bar.setVisible(False)
        self.refresh_current_scene()
        self.update_roi_status(f"匹配完成，新增 {match_total} 个 ROI")

    def match_single_image(self, img_path, settings):
        if not os.path.exists(img_path): return []
        target_img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if target_img is None: return []

        existing_rois = self.all_rois.get(img_path, [])
        chip_id = get_chip_index(img_path)
        if chip_id is None: chip_id = 0

        results = halcon_api.match_template(target_img, self.template_path, settings)
        final_rects = []
        tw, th = self.template_w, self.template_h

        for res in results:
            cx, cy = res['col'], res['row']
            x = int(cx - tw / 2);
            y = int(cy - th / 2)
            new_r = {'x': x, 'y': y, 'w': tw, 'h': th, 'para': chip_id}

            keep = True
            for r in final_rects:
                if is_rect_overlap(new_r, r): keep = False; break
            if keep:
                for ex in existing_rois:
                    if is_rect_overlap(new_r, ex): keep = False; break
            if keep: final_rects.append(new_r)
        return final_rects

    def prepare_template_from_item(self, roi_item):
        if self.current_img_idx == -1: return
        img_path = self.image_paths[self.current_img_idx]
        if not os.path.exists(img_path): return
        img_np = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img_np is None: return
        roi_rect = roi_item.rect();
        roi_pos = roi_item.pos()
        x = int(roi_pos.x() + roi_rect.x());
        y = int(roi_pos.y() + roi_rect.y())
        w = int(roi_rect.width());
        h = int(roi_rect.height())
        h_img, w_img = img_np.shape[:2]
        x = max(0, x);
        y = max(0, y);
        w = min(w, w_img - x);
        h = min(h, h_img - y)
        if w <= 0 or h <= 0: return
        template_img = img_np[y:y + h, x:x + w].copy()
        if not halcon_api.dll:
            QMessageBox.critical(self, "Error", "Halcon DLL 未加载，无法生成模板。")
            return
        success = halcon_api.make_template(template_img, self.template_path)
        if not success:
            QMessageBox.critical(self, "Error", "Halcon 模板生成失败。")
            return
        self.template_w = w;
        self.template_h = h
        dlg = MatchConfigDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            self.run_matching_process(dlg.get_settings())

    def clear_current_rois_on_canvas(self):
        for item in [i for i in self.canvas.scene.items() if isinstance(i, ResizableRectItem)]:
            self.canvas.scene.removeItem(item)
        self.update_roi_status()

    def on_para_changed(self, val):
        self.canvas.roi_para_counter = val
        for item in self.canvas.scene.selectedItems():
            if isinstance(item, ResizableRectItem): item.roi_para = val



    def copy_to_any_image(self):
        if self.current_img_idx == -1: return
        self.save_current_rois_to_memory()
        curr_path = self.image_paths[self.current_img_idx]
        tid, ok = QInputDialog.getInt(self, "复制到", "目标 Chip ID:", 0, 0, 999999)
        if not ok: return
        tgt_path = next((p for p in self.image_paths if get_chip_index(p) == tid), None)
        if not tgt_path: return QMessageBox.warning(self, "Err", "未找到图片")
        tr = self.all_rois.get(tgt_path, []);
        cnt = 0
        for r in self.all_rois.get(curr_path, []):
            nr = r.copy();
            nr['para'] = tid
            if not any(is_rect_overlap(nr, ex) for ex in tr): tr.append(nr); cnt += 1
        self.all_rois[tgt_path] = tr;
        QMessageBox.information(self, "OK", f"复制 {cnt} 个")

    def copy_previous_roi(self):
        if self.current_img_idx <= 0: return
        self.save_current_rois_to_memory()
        prev_rois = self.all_rois.get(self.image_paths[self.current_img_idx - 1], [])
        curr_para = self.canvas.roi_para_counter;
        curr_path = self.image_paths[self.current_img_idx]
        ex_rois = self.all_rois.get(curr_path, []);
        cnt = 0
        for r in prev_rois:
            nr = {'x': r['x'], 'y': r['y'], 'w': r['w'], 'h': r['h'], 'para': curr_para}
            if not any(is_rect_overlap(nr, ex) for ex in ex_rois):
                self.canvas.add_roi(r['x'], r['y'], r['w'], r['h'], curr_para);
                ex_rois.append(nr);
                cnt += 1
        self.update_roi_status(f"复制 {cnt} 个")

    def load_images(self):
        d = self.settings.value("last_image_dir", "");
        fs, _ = QFileDialog.getOpenFileNames(self, "图", d, "Img (*.png *.jpg *.bmp)")
        if fs:
            self.settings.setValue("last_image_dir", os.path.dirname(fs[0]))
            for f in fs:
                if f not in self.image_paths: self.image_paths.append(f); self.list_widget.addItem(
                    os.path.basename(f)); self.all_rois.setdefault(f, [])
            if self.list_widget.count() > 0 and self.current_img_idx == -1: self.list_widget.setCurrentRow(
                0); self.on_list_clicked()

    # ------------------ XML 处理 (修复版) ------------------

    def load_xml(self):
        f, _ = QFileDialog.getOpenFileName(self, "选择XML", self.settings.value("last_xml_dir", ""),
                                           "XML Files (*.xml)")
        if not f: return
        self.settings.setValue("last_xml_dir", os.path.dirname(f))

        try:
            # 1. 保存完整的 XML 树结构，以便后续保存时复用其他节点
            self.xml_tree = ET.parse(f)
            root = self.xml_tree.getroot()

            roi_rects = root.find("RoiRects")

            # 建立图片索引
            index_map = {get_chip_index(p): p for p in self.image_paths if get_chip_index(p) is not None}

            # 初始化所有图片的 ROI 列表
            self.all_rois = {p: [] for p in self.image_paths}

            cnt = 0
            if roi_rects is not None:
                for info in roi_rects.findall("RoiRectInfo"):
                    para_node = info.find("RoiPara")
                    if para_node is None: continue
                    para_val = int(para_node.text)

                    if para_val in index_map:
                        target_path = index_map[para_val]
                        width_node = info.find("Width")
                        height_node = info.find("Height")
                        pt_node = info.find("RoiStartPoint")

                        if width_node is not None and height_node is not None and pt_node is not None:
                            w = int(width_node.text)
                            h = int(height_node.text)
                            x = int(pt_node.find("X").text)
                            y = int(pt_node.find("Y").text)

                            self.all_rois[target_path].append({
                                'x': x, 'y': y, 'w': w, 'h': h, 'para': para_val
                            })
                            cnt += 1

            self.refresh_current_scene()
            QMessageBox.information(self, "Info", f"XML加载成功，读取到 {cnt} 个ROI。")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"XML 解析失败: {str(e)}")
            self.xml_tree = None  # 加载失败则置空

    def save_xml(self):
        self.save_current_rois_to_memory()
        f, _ = QFileDialog.getSaveFileName(self, "保存", self.settings.value("last_xml_dir", ""), "XML (*.xml)")
        if not f: return
        self.settings.setValue("last_xml_dir", os.path.dirname(f))

        # 1. 确定根节点
        if self.xml_tree:
            # 如果加载过 XML，复用它以保留其他节点
            root = self.xml_tree.getroot()
        else:
            # 如果没加载过，新建标准结构
            root = ET.Element("InspectionPara")
            ET.SubElement(root, "ParaName").text = "InspectionPara"

        # 2. 定位或创建 RoiRects 节点
        roi_rects_node = root.find("RoiRects")
        if roi_rects_node is not None:
            # 关键：清空旧的 ROI 数据，但保留节点本身
            roi_rects_node.clear()
            # clear() 会把 tail 也清空，导致格式乱掉，这里不用管，后面统一 indent
        else:
            roi_rects_node = ET.SubElement(root, "RoiRects")

        # 3. 写入新的 ROI 数据
        cnt = 0
        # 为了保证 XML 中的顺序（例如按 ChipID 排序），我们可以先收集所有 ROI
        # 这里为了简单，直接按 image_paths 的顺序写入
        for p in self.image_paths:
            rois = self.all_rois.get(p, [])
            for r in rois:
                cnt += 1
                info = ET.SubElement(roi_rects_node, "RoiRectInfo")
                ET.SubElement(info, "Width").text = str(r['w'])
                ET.SubElement(info, "Height").text = str(r['h'])
                ET.SubElement(info, "RoiPara").text = str(r['para'])

                pt = ET.SubElement(info, "RoiStartPoint")
                ET.SubElement(pt, "X").text = str(r['x'])
                ET.SubElement(pt, "Y").text = str(r['y'])

                ET.SubElement(info, "IsUsedRoi").text = "true"

        # 4. 格式化缩进 (修复一行的问题)
        self.indent(root)

        # 5. 保存文件
        tree = ET.ElementTree(root)
        try:
            tree.write(f, encoding="utf-8", xml_declaration=True)
            QMessageBox.information(self, "OK", f"保存成功！共写入 {cnt} 个ROI。")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"保存失败: {str(e)}")

    def indent(self, elem, level=0):
        """XML 格式化缩进函数"""
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self.indent(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    def save_current_rois_to_memory(self):
        if self.current_img_idx == -1: return
        p = self.image_paths[self.current_img_idx]
        self.all_rois[p] = sorted([{'x': int(i.scenePos().x() + i.rect().x()),
                                    'y': int(i.scenePos().y() + i.rect().y()), 'w': int(i.rect().width()),
                                    'h': int(i.rect().height()), 'para': i.roi_para} for i in self.canvas.scene.items()
                                   if isinstance(i, ResizableRectItem)], key=lambda k: k['y'])

    def clear_images(self):
        self.canvas.scene.clear(); self.list_widget.clear(); self.image_paths = []; self.all_rois = {}; self.current_img_idx = -1; self.update_roi_status()

    def delete_current_image(self):
        r = self.list_widget.currentRow();
        if r >= 0: p = self.image_paths.pop(r); self.all_rois.pop(p, None); self.list_widget.takeItem(
            r); self.canvas.scene.clear(); self.current_img_idx = -1;
        if self.list_widget.count() > 0: self.list_widget.setCurrentRow(
            min(r, self.list_widget.count() - 1)); self.on_list_clicked()

    def move_image_up(self):
        r = self.list_widget.currentRow();
        if r > 0: self.save_current_rois_to_memory(); self.image_paths[r], self.image_paths[r - 1] = self.image_paths[
            r - 1], self.image_paths[r]; i = self.list_widget.takeItem(r); self.list_widget.insertItem(r - 1,
                                                                                                       i); self.list_widget.setCurrentRow(
            r - 1); self.on_list_clicked()

    def move_image_down(self):
        r = self.list_widget.currentRow();
        if r >= 0 and r < self.list_widget.count() - 1: self.save_current_rois_to_memory(); self.image_paths[r], \
        self.image_paths[r + 1] = self.image_paths[r + 1], self.image_paths[r]; i = self.list_widget.takeItem(
            r); self.list_widget.insertItem(r + 1, i); self.list_widget.setCurrentRow(r + 1); self.on_list_clicked()

    def show_list_context_menu(self, pos):
        if self.list_widget.itemAt(pos):
            m = QMenu();
            a1 = m.addAction("删");
            a2 = m.addAction("上");
            a3 = m.addAction("下");
            res = m.exec_(self.list_widget.mapToGlobal(pos))
            if res == a1:
                self.delete_current_image()
            elif res == a2:
                self.move_image_up()
            elif res == a3:
                self.move_image_down()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())